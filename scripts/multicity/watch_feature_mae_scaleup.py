#!/usr/bin/env python3
"""Low-overhead safety watcher for the Feature-MAE scale-up runner."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import signal
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_ROOT = Path("outputs/experiments/dinov3_multicity/feature_mae_scaleup")
TERMINAL_STATES = {"completed", "failed", "blocked_resources"}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _memory_snapshot() -> dict[str, float]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value, *_ = line.split()
        values[name.rstrip(":")] = int(value) / 2**20
    return {
        "total_gib": values["MemTotal"],
        "available_gib": values["MemAvailable"],
        "swap_total_gib": values["SwapTotal"],
        "swap_free_gib": values["SwapFree"],
    }


def _cpu_times() -> tuple[int, int]:
    fields = [
        int(value)
        for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    ]
    idle = fields[3] + fields[4]
    return sum(fields), idle


def _cpu_utilization(previous: tuple[int, int], current: tuple[int, int]) -> float:
    total = current[0] - previous[0]
    idle = current[1] - previous[1]
    return 0.0 if total <= 0 else 100.0 * (total - idle) / total


def _cpu_package_temperature_c() -> float | None:
    readings = []
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            zone_type = (zone / "type").read_text().strip()
            if zone_type not in {"x86_pkg_temp", "coretemp", "cpu-thermal"}:
                continue
            readings.append(float((zone / "temp").read_text().strip()) / 1000.0)
        except (OSError, ValueError):
            continue
    return max(readings) if readings else None


def _gpu_snapshot() -> dict[str, float]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free,memory.total,temperature.gpu,"
            "utilization.gpu,utilization.memory,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    values = [float(value.strip()) for value in output.splitlines()[0].split(",")]
    names = (
        "used_mib",
        "free_mib",
        "total_mib",
        "temperature_c",
        "utilization_pct",
        "memory_utilization_pct",
        "power_draw_w",
        "power_limit_w",
    )
    return dict(zip(names, values))


def _find_pid(module_name: str) -> int | None:
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (OSError, UnicodeDecodeError):
            continue
        if module_name in command:
            return int(proc.name)
    return None


def _current_experiment(root: Path) -> str | None:
    try:
        with (root / "experiment_registry.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") == "running":
                    return row["experiment_id"]
    except OSError:
        pass
    return None


def _file_snapshot(path: Path) -> dict:
    try:
        stat = path.stat()
        return {
            "path": str(path),
            "exists": True,
            "size_bytes": stat.st_size,
            "modified_utc": datetime.fromtimestamp(
                stat.st_mtime, timezone.utc
            ).isoformat(),
            "age_seconds": max(0.0, time.time() - stat.st_mtime),
        }
    except OSError:
        return {"path": str(path), "exists": False}


def _process_cpu_pct(pid: int | None) -> float | None:
    if pid is None:
        return None
    try:
        output = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "%cpu="], text=True, timeout=5
        ).strip()
        return float(output) if output else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _process_snapshot(pid: int | None) -> dict | None:
    if pid is None:
        return None
    try:
        status = {}
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "Threads:")):
                key, value, *_ = line.split()
                status[key.rstrip(":")] = int(value)
        return {
            "pid": pid,
            "cpu_pct": _process_cpu_pct(pid),
            "rss_gib": status.get("VmRSS", 0) / 2**20,
            "threads": status.get("Threads"),
            "cpu_affinity": sorted(os.sched_getaffinity(pid)),
        }
    except (OSError, ProcessLookupError):
        return None


def _ensure_full_cpu_affinity(pid: int | None, full_cpu_set: set[int]) -> bool:
    if pid is None:
        return False
    try:
        if os.sched_getaffinity(pid) == full_cpu_set:
            return False
        os.sched_setaffinity(pid, full_cpu_set)
        return True
    except (OSError, ProcessLookupError):
        return False


def _terminate(pid: int, log) -> None:
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if not Path(f"/proc/{pid}").exists():
            return
        time.sleep(1)
    print(
        f"{_utc()} trainer {pid} ignored SIGTERM; sending SIGKILL",
        file=log,
        flush=True,
    )
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--ram-min-gib", type=float, default=4.0)
    parser.add_argument("--gpu-temp-max", type=float, default=84.0)
    parser.add_argument("--cpu-temp-max", type=float, default=90.0)
    parser.add_argument("--gpu-free-min-mib", type=float, default=512.0)
    parser.add_argument("--disk-free-min-gib", type=float, default=500.0)
    parser.add_argument("--consecutive-critical", type=int, default=2)
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    lock = (args.root / "watcher.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another Feature-MAE watcher is already active") from error

    state_path = args.root / "watcher_state.json"
    runner_state_path = args.root / "runner_state.json"
    full_cpu_set = set(range(os.cpu_count() or 1))
    os.sched_setaffinity(0, full_cpu_set)
    allowed_cpus = sorted(full_cpu_set)
    critical_streak = 0
    last_reasons: tuple[str, ...] = ()
    previous_cpu_times = _cpu_times()

    with (args.root / "watcher.log").open("a", buffering=1) as log:
        print(
            f"{_utc()} watcher started; interval={args.interval:.0f}s "
            f"cpus={allowed_cpus}",
            file=log,
        )
        while True:
            runner_state = {}
            try:
                runner_state = json.loads(runner_state_path.read_text())
            except (OSError, json.JSONDecodeError):
                pass
            status = runner_state.get("status", "unknown")
            runner_pid = _find_pid("scripts.multicity.run_feature_mae_scaleup")
            trainer_pid = _find_pid("scripts.multicity.train_feature_mae")
            experiment_id = _current_experiment(args.root)
            if _ensure_full_cpu_affinity(runner_pid, full_cpu_set):
                print(
                    f"{_utc()} removed CPU affinity limit from runner {runner_pid}",
                    file=log,
                )
            if _ensure_full_cpu_affinity(trainer_pid, full_cpu_set):
                print(
                    f"{_utc()} removed CPU affinity limit from trainer {trainer_pid}",
                    file=log,
                )

            try:
                gpu = _gpu_snapshot()
                gpu_error = None
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                gpu = {}
                gpu_error = str(error)
            memory = _memory_snapshot()
            cpu_temperature = _cpu_package_temperature_c()
            current_cpu_times = _cpu_times()
            cpu_utilization = _cpu_utilization(previous_cpu_times, current_cpu_times)
            previous_cpu_times = current_cpu_times
            load1 = os.getloadavg()[0]
            runner_process = _process_snapshot(runner_pid)
            trainer_process = _process_snapshot(trainer_pid)
            disk = shutil.disk_usage(args.root)
            disk_free_gib = disk.free / 2**30
            disk_used_pct = 100.0 * disk.used / disk.total
            run_dir = args.root / "runs" / experiment_id if experiment_id else None
            artifacts = {}
            if run_dir is not None:
                artifacts = {
                    name: _file_snapshot(run_dir / name)
                    for name in (
                        "checkpoint_last.pt",
                        "model_best.pt",
                        "training.log",
                        "training_report.json",
                    )
                }

            reasons = []
            if memory["available_gib"] < args.ram_min_gib:
                reasons.append(
                    f"RAM {memory['available_gib']:.1f} GiB < "
                    f"{args.ram_min_gib:.1f} GiB"
                )
            if gpu and gpu["temperature_c"] >= args.gpu_temp_max:
                reasons.append(
                    f"GPU temperature {gpu['temperature_c']:.0f} C >= "
                    f"{args.gpu_temp_max:.0f} C"
                )
            if gpu and gpu["free_mib"] < args.gpu_free_min_mib:
                reasons.append(
                    f"GPU free memory {gpu['free_mib']:.0f} MiB < "
                    f"{args.gpu_free_min_mib:.0f} MiB"
                )
            if cpu_temperature is not None and cpu_temperature >= args.cpu_temp_max:
                reasons.append(
                    f"CPU temperature {cpu_temperature:.0f} C >= "
                    f"{args.cpu_temp_max:.0f} C"
                )
            if disk_free_gib < args.disk_free_min_gib:
                reasons.append(
                    f"disk free {disk_free_gib:.1f} GiB < "
                    f"{args.disk_free_min_gib:.1f} GiB"
                )

            reason_tuple = tuple(reasons)
            critical_streak = critical_streak + 1 if reasons else 0
            snapshot = {
                "time_utc": _utc(),
                "status": "critical" if reasons else "safe",
                "runner_status": status,
                "runner_pid": runner_pid,
                "trainer_pid": trainer_pid,
                "experiment_id": experiment_id,
                "allowed_cpus": allowed_cpus,
                "load_1m": load1,
                "cpu_utilization_pct": cpu_utilization,
                "cpu_temperature_c": cpu_temperature,
                "runner_process": runner_process,
                "trainer_process": trainer_process,
                "memory": memory,
                "disk_free_gib": disk_free_gib,
                "disk_used_pct": disk_used_pct,
                "artifacts": artifacts,
                "gpu": gpu,
                "gpu_query_error": gpu_error,
                "critical_reasons": reasons,
                "critical_streak": critical_streak,
            }
            _atomic_json(state_path, snapshot)

            if reason_tuple != last_reasons:
                if reasons:
                    print(f"{_utc()} CRITICAL: {'; '.join(reasons)}", file=log)
                elif last_reasons:
                    print(f"{_utc()} resources recovered to safe range", file=log)
                last_reasons = reason_tuple

            if reasons and critical_streak >= args.consecutive_critical and trainer_pid:
                print(
                    f"{_utc()} SAFETY_STOP trainer={trainer_pid}: "
                    f"{'; '.join(reasons)}",
                    file=log,
                    flush=True,
                )
                _terminate(trainer_pid, log)
                return

            if status in TERMINAL_STATES and runner_pid is None and trainer_pid is None:
                print(f"{_utc()} watcher exiting on runner status={status}", file=log)
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
