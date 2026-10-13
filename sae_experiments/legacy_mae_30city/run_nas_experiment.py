"""Resume four-GPU extraction, train BatchTopK W4096/K32, then derive figures."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .nas_batch import CITIES, OUTPUT
from .train import write_json


def run_stage(module: str, args: list[str], output: Path) -> None:
    command = [sys.executable, "-m", f"sae_experiments.legacy_mae_30city.{module}", *args]
    log = output / f"{module}_runner.log"
    write_json(output / "runner_status.json", {"stage": module, "status": "running", "log": str(log)})
    print("stage", module, flush=True)
    with log.open("a") as stream:
        stream.write("\n$ " + " ".join(command) + "\n")
        stream.flush()
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def completed_cities(output: Path) -> int:
    root = output / "tokens" / "patch_tokens"
    return sum((root / city.replace("/", "__") / "tokens.f16.npy").is_file()
               and (root / city.replace("/", "__") / "token_report.json").is_file()
               for city in CITIES)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--gpu-slices", default="1:0:8,0:8:15,2:15:23,3:23:30")
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--train-device", default="cuda:1")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--wait-existing-extraction", action="store_true",
                   help="observe already-running GPU extractors instead of starting duplicates")
    args = p.parse_args()
    if min(args.epochs, args.poll_seconds, args.batch_size, args.workers) < 1:
        p.error("counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_root = args.output / "manifests"
    if not (manifest_root / "batch.json").is_file():
        run_stage("nas_batch", ["build", "--output", str(args.output)], args.output)
    slices = []
    for part in args.gpu_slices.split(","):
        gpu, start, stop = map(int, part.split(":"))
        if not 0 <= start < stop <= len(CITIES):
            p.error("invalid city slice")
        slices.append((gpu, start, stop))
    if sorted(i for _, a, b in slices for i in range(a, b)) != list(range(len(CITIES))):
        p.error("GPU slices must cover every city exactly once")
    if completed_cities(args.output) < len(CITIES) and args.wait_existing_extraction:
        from .train import digest
        worker_start = time.monotonic()
        while completed_cities(args.output) < len(CITIES):
            for city in CITIES:
                slug = city.replace("/", "__")
                manifest = manifest_root / (slug + ".parquet")
                token_dir = args.output / "tokens" / "patch_tokens" / slug
                token = token_dir / "tokens.f16.npy"
                progress = token_dir / "tokens.progress.json"
                report = token_dir / "token_report.json"
                if report.is_file():
                    if json.loads(report.read_text()).get("manifest_sha256") != digest(manifest):
                        raise ValueError(f"Manifest/token mismatch: {city}")
                if progress.is_file() and json.loads(progress.read_text()).get("manifest_sha256") != digest(manifest):
                    raise ValueError(f"Manifest/partial mismatch: {city}")
            count = completed_cities(args.output)
            write_json(args.output / "runner_status.json", {"stage": "extract", "status": "running",
                       "complete_cities": count, "total_cities": len(CITIES),
                       "elapsed_seconds": round(time.monotonic() - worker_start)})
            print("waiting for token extraction", count, "/", len(CITIES), flush=True)
            time.sleep(args.poll_seconds)
    elif completed_cities(args.output) < len(CITIES):
        processes: list[tuple[int, subprocess.Popen, Path]] = []
        for gpu, start, stop in slices:
            log = args.output / f"extract_gpu{gpu}_runner.log"
            command = [sys.executable, "-m", "sae_experiments.legacy_mae_30city.nas_batch",
                       "extract", "--output", str(args.output), "--device", f"cuda:{gpu}",
                       "--city-start", str(start), "--city-stop", str(stop),
                       "--batch-size", str(args.batch_size), "--workers", str(args.workers)]
            with log.open("a") as stream:
                stream.write("\n$ " + " ".join(command) + "\n")
                stream.flush()
                proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            processes.append((gpu, proc, log))
        write_json(args.output / "runner_status.json", {"stage": "extract", "status": "running",
                   "complete_cities": completed_cities(args.output),
                   "worker_pids": [proc.pid for _, proc, _ in processes]})
        try:
            while processes:
                still = []
                for gpu, proc, log in processes:
                    rc = proc.poll()
                    if rc is None:
                        still.append((gpu, proc, log))
                    elif rc:
                        raise RuntimeError(f"GPU {gpu} extraction exited {rc}; inspect {log}")
                    else:
                        print("extraction worker complete gpu", gpu, flush=True)
                processes = still
                if processes:
                    time.sleep(args.poll_seconds)
        except BaseException:
            for _, proc, _ in processes:
                proc.terminate()
            raise
    count = completed_cities(args.output)
    if count != len(CITIES):
        raise RuntimeError(f"Only {count}/30 token caches complete")
    quality_dir = args.output / "quality"
    needed = [city for city in CITIES if not (quality_dir / (city.replace("/", "__") + ".json")).is_file()]
    if needed:
        print("screening black and tunnel panoramas", len(needed), "cities", flush=True)
        screen_jobs = []
        for worker_index, (_, start, stop) in enumerate(slices):
            command = [sys.executable, "-m", "sae_experiments.legacy_mae_30city.nas_batch",
                       "screen", "--output", str(args.output), "--city-start", str(start),
                       "--city-stop", str(stop), "--workers", str(args.workers)]
            log = args.output / f"screen_worker{worker_index}.log"
            with log.open("a") as stream:
                stream.write("\n$ " + " ".join(command) + "\n")
                stream.flush()
                screen_jobs.append((subprocess.Popen(command, stdout=stream,
                                                      stderr=subprocess.STDOUT), log))
        write_json(args.output / "runner_status.json", {"stage": "quality", "status": "running",
                                                       "worker_pids": [job.pid for job, _ in screen_jobs]})
        for job, log in screen_jobs:
            rc = job.wait()
            if rc:
                raise RuntimeError(f"Quality screening exited {rc}; inspect {log}")
    if sum((quality_dir / (city.replace("/", "__") + ".json")).is_file() for city in CITIES) != len(CITIES):
        raise RuntimeError("Some city quality screening outputs are absent")
    common = ["--output", str(args.output), "--manifest-root", str(manifest_root),
              "--token-root", str(args.output / "tokens")]
    run_stage("train", [*common, "--device", args.train_device, "--epochs", str(args.epochs)], args.output)
    checkpoint = str(args.output / "model_best.pt")
    run_stage("cluster", ["--output", str(args.output), "--checkpoint", checkpoint], args.output)
    run_stage("infer", [*common, "--device", args.train_device, "--checkpoint", checkpoint], args.output)
    run_stage("figures", ["--output", str(args.output), "--manifest-root", str(manifest_root),
                          "--figures", str(args.output / "figures")], args.output)
    run_stage("extra_figures", ["--output", str(args.output)], args.output)
    run_stage("visual_figures", ["--output", str(args.output), "--device", args.train_device], args.output)
    run_stage("pair_figures", ["--output", str(args.output), "--device", "cpu"], args.output)
    run_stage("spatial_figures", ["--output", str(args.output)], args.output)
    batch = json.loads((manifest_root / "batch.json").read_text())
    inference = json.loads((args.output / "inference_report.json").read_text())
    inventory = json.loads((args.output / "figure_inventory.json").read_text())
    write_json(args.output / "runner_status.json", {"stage": "complete", "status": "complete",
                                                     "manifest_images": batch["images"],
                                                     "quality_passed_images": inference["images"],
                                                     "sae_analogue_figures": inventory["recomputed_count"]})


if __name__ == "__main__":
    main()
