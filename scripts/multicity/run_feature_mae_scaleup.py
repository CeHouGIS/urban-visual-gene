#!/usr/bin/env python3
"""Run the 21-task Feature-MAE scale-up suite sequentially and resumably."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_ROOT = Path(
    "outputs/experiments/dinov3_multicity/feature_mae_scaleup"
)
SAMPLE_SIZES = (500, 1000, 2000, 4000, 8000, 10000)
WIDTHS = (128, 256, 512, 1024)
MICROBATCH = {128: 240, 256: 240, 512: 240, 1024: 60}
EXCLUDED_CPUS = {8, 9}


def _allowed_cpus() -> set[int]:
    """Return the current affinity mask with the reserved CPUs removed."""
    current = set(os.sched_getaffinity(0))
    allowed = current.difference(EXCLUDED_CPUS)
    if not allowed:
        raise RuntimeError(
            f"no CPUs remain after excluding reserved CPUs {sorted(EXCLUDED_CPUS)}"
        )
    return allowed


def _configuration_order() -> list[dict]:
    values = []

    def add(sample_size: int, width: int, seed: int, group: str) -> None:
        experiment_id = f"scale_n{sample_size:05d}_w{width:04d}_s{seed}"
        if not any(item["experiment_id"] == experiment_id for item in values):
            values.append(
                {
                    "experiment_id": experiment_id,
                    "sample_size": sample_size,
                    "width": width,
                    "seed": seed,
                    "group": group,
                }
            )

    add(10000, 512, 42, "reference")
    for sample_size in SAMPLE_SIZES:
        add(sample_size, 512, 42, "sample_curve")
    for width in (128, 256, 1024):
        add(10000, width, 42, "width_curve")
    for seed in (43, 44):
        for sample_size in (1000, 4000, 10000):
            add(sample_size, 512, seed, "sample_replicate")
        for width in (128, 256, 1024):
            add(10000, width, seed, "width_replicate")
    if len(values) != 21:
        raise AssertionError(f"expected 21 unique tasks, got {len(values)}")
    return values


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _available_ram_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    raise RuntimeError("MemAvailable not found")


def _gpu_free_mib() -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return int(output.strip().splitlines()[0])


def _gpu_temperature_c() -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi", "--query-gpu=temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return int(output.strip().splitlines()[0])


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _write_registry(path: Path, rows: list[dict]) -> None:
    fields = [
        "experiment_id", "group", "sample_size", "width", "seed", "status",
        "started_utc", "finished_utc", "return_code", "resume_count", "output_dir",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--epoch-images-per-city", type=int, default=10000)
    parser.add_argument("--images-per-city", type=int, default=16)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()

    allowed_cpus = _allowed_cpus()
    os.sched_setaffinity(0, allowed_cpus)

    args.root.mkdir(parents=True, exist_ok=True)
    lock_handle = (args.root / "scaleup.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another scale-up runner already owns the lock") from error

    configurations = _configuration_order()
    if args.only:
        requested = set(args.only)
        configurations = [c for c in configurations if c["experiment_id"] in requested]
        missing = requested.difference(c["experiment_id"] for c in configurations)
        if missing:
            raise ValueError(f"unknown experiment IDs: {sorted(missing)}")

    registry_path = args.root / "experiment_registry.csv"
    old = {}
    if registry_path.exists():
        with registry_path.open(newline="") as handle:
            old = {row["experiment_id"]: row for row in csv.DictReader(handle)}
    rows = []
    for config in configurations:
        prior = old.get(config["experiment_id"], {})
        rows.append(
            {
                **config,
                "status": prior.get("status", "planned"),
                "started_utc": prior.get("started_utc", ""),
                "finished_utc": prior.get("finished_utc", ""),
                "return_code": prior.get("return_code", ""),
                "resume_count": prior.get("resume_count", "0"),
                "output_dir": str((args.root / "runs" / config["experiment_id"]).resolve()),
            }
        )
    _write_registry(registry_path, rows)
    _atomic_json(
        args.root / "runner_state.json",
        {"status": "running", "started_utc": _utc(), "tasks": len(rows)},
    )

    log_path = args.root / "runner.log"
    with log_path.open("a", buffering=1) as runner_log:
        for index, (config, row) in enumerate(zip(configurations, rows), 1):
            output = Path(row["output_dir"])
            report = output / "training_report.json"
            if report.is_file():
                row["status"] = "completed"
                _write_registry(registry_path, rows)
                print(f"[{index}/{len(rows)}] skip complete {config['experiment_id']}", file=runner_log)
                continue
            ram = _available_ram_gib()
            disk = shutil.disk_usage(args.root).free / 2**30
            gpu = _gpu_free_mib()
            if ram < 8 or disk < 500 or gpu < 10000:
                message = f"unsafe resources: ram={ram:.1f}GiB disk={disk:.1f}GiB gpu={gpu}MiB"
                row["status"] = "blocked_resources"
                _write_registry(registry_path, rows)
                _atomic_json(
                    args.root / "runner_state.json",
                    {"status": "blocked_resources", "time_utc": _utc(), "message": message},
                )
                raise RuntimeError(message)

            output.mkdir(parents=True, exist_ok=True)
            row["status"] = "running"
            row["started_utc"] = _utc()
            row["resume_count"] = str(int(row["resume_count"] or 0) + int((output / "checkpoint_last.pt").exists()))
            _write_registry(registry_path, rows)
            command = [
                sys.executable, "-m", "scripts.multicity.train_feature_mae",
                "--output-dir", str(output),
                "--train-manifest-root", str(args.root / "splits" / f"train_n{config['sample_size']:05d}"),
                "--validation-manifest-root", str(args.root / "splits" / "validation"),
                "--epoch-images-per-city", str(args.epoch_images_per_city),
                "--width", str(config["width"]),
                "--mask-ratio", "0.75",
                "--encoder-layers", "4",
                "--decoder-width", "256",
                "--decoder-layers", "2",
                "--epochs", str(args.epochs),
                "--images-per-city", str(args.images_per_city),
                "--microbatch", str(MICROBATCH[config["width"]]),
                "--prefetch-batches", "1",
                "--gradient-accumulation", "1",
                "--seed", str(config["seed"]),
            ]
            print(f"[{index}/{len(rows)}] start {config['experiment_id']} at {_utc()}", file=runner_log)
            env = os.environ.copy()
            thread_count = str(len(allowed_cpus))
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": "0",
                    "OMP_NUM_THREADS": thread_count,
                    "MKL_NUM_THREADS": thread_count,
                    "OPENBLAS_NUM_THREADS": thread_count,
                }
            )
            safety_stop = ""
            with (output / "training.log").open("a", buffering=1) as training_log:
                process = subprocess.Popen(
                    command, stdout=training_log, stderr=subprocess.STDOUT, env=env
                )
                while process.poll() is None:
                    time.sleep(30)
                    ram_now = _available_ram_gib()
                    temperature = _gpu_temperature_c()
                    if ram_now < 4:
                        safety_stop = f"available RAM fell to {ram_now:.1f} GiB"
                    elif temperature >= 84:
                        safety_stop = f"GPU temperature reached {temperature} C"
                    if safety_stop:
                        print(
                            f"SAFETY_STOP {config['experiment_id']}: {safety_stop}",
                            file=runner_log,
                        )
                        process.terminate()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        break
                return_code = process.wait()
            row["finished_utc"] = _utc()
            row["return_code"] = str(return_code)
            row["status"] = "completed" if report.is_file() and return_code == 0 else "failed"
            _write_registry(registry_path, rows)
            if row["status"] != "completed":
                _atomic_json(
                    args.root / "runner_state.json",
                    {
                        "status": "failed", "time_utc": _utc(),
                        "experiment_id": config["experiment_id"],
                        "return_code": return_code,
                        "safety_stop": safety_stop,
                    },
                )
                raise RuntimeError(f"failed {config['experiment_id']}: {return_code}")
            print(f"[{index}/{len(rows)}] complete {config['experiment_id']} at {_utc()}", file=runner_log)

    _atomic_json(
        args.root / "runner_state.json",
        {"status": "completed", "finished_utc": _utc(), "tasks": len(rows)},
    )


if __name__ == "__main__":
    main()
