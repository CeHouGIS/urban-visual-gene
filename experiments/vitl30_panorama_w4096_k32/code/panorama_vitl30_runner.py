"""Continue a completed cache-backed SAE run through analysis and figures."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from sae_experiments.io_utils import write_json
from sae_experiments.panorama_vitl30 import OUTPUT
from sae_experiments.panorama_vitl30_cache import names

ANALYSIS = "sae_experiments.panorama_vitl30_analysis"
FIGURES = "sae_experiments.panorama_vitl30_figures"


def command(module: str, output: Path, *extra: str) -> list[str]:
    return [sys.executable, "-m", module, *extra, "--output", str(output)]


def active_inference(output: Path) -> list[int]:
    """Find inference processes for this output, including absolute Python paths."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().decode().strip("\0").split("\0")
            pos = argv.index("-m")
            if argv[pos + 1:pos + 3] != [ANALYSIS, "infer"]:
                continue
            destination = Path(argv[argv.index("--output") + 1]) if "--output" in argv else OUTPUT
            if destination.resolve() == output.resolve():
                found.append(int(entry.name))
        except (OSError, UnicodeError, ValueError, IndexError):
            continue
    return sorted(found)


def inference_progress(output: Path) -> dict:
    audit = json.loads((output / "cached_feature_audit.json").read_text())
    total = sum(row["panoramas"] for row in audit["cities"])
    done = cities_done = 0
    for row in audit["cities"]:
        folder = output / "analysis/cities" / row["city"].replace("/", "__")
        state = folder / "progress.json"
        if state.is_file():
            completed = int(json.loads(state.read_text())["completed"])
            done += min(completed, row["panoramas"])
            cities_done += int(completed == row["panoramas"] and
                               (folder / "panorama_group_share.f32.npy").is_file())
    return {"completed_panoramas": done, "total_panoramas": total,
            "completed_cities": cities_done, "total_cities": len(audit["cities"])}


def status(output: Path, stage: str, state: str = "running", **extra):
    write_json(output / "runner_status.json", {
        "stage": stage, "state": state, "pid": os.getpid(),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "training": "complete", "features_reused": True,
        "extraction_enabled": False, **extra,
    })


def run_stage(output: Path, stage: str, cmd: list[str], attempts: int = 3):
    for attempt in range(1, attempts + 1):
        status(output, stage, attempt=attempt)
        with (output / f"runner_{stage}.log").open("a") as log:
            log.write(f"\nAttempt {attempt}: {' '.join(cmd)}\n")
            log.flush()
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        if code == 0:
            return
        status(output, stage, "retrying", attempt=attempt, exit_code=code)
        if attempt < attempts:
            time.sleep(20)
    raise RuntimeError(f"{stage} failed {attempts} times; see runner_{stage}.log")


def resume_inference(args):
    # Adopt existing jobs before starting resumable workers to prevent duplicate writers.
    while pids := active_inference(args.output):
        status(args.output, "infer", existing_pids=pids, **inference_progress(args.output))
        time.sleep(args.poll_seconds)
    count = len(names(args.output))
    for attempt in range(1, 4):
        jobs, handles = [], []
        try:
            for i, device in enumerate(args.devices):
                start, stop = i * count // len(args.devices), (i + 1) * count // len(args.devices)
                cmd = command(ANALYSIS, args.output, "infer", "--city-start", str(start),
                              "--city-stop", str(stop), "--device", device)
                log = (args.output / f"runner_infer_{i}.log").open("a")
                handles.append(log)
                log.write(f"\nAttempt {attempt}: {' '.join(cmd)}\n")
                log.flush()
                jobs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
            while any(job.poll() is None for job in jobs):
                status(args.output, "infer", attempt=attempt, devices=args.devices,
                       **inference_progress(args.output))
                time.sleep(args.poll_seconds)
            codes = [job.wait() for job in jobs]
        finally:
            for handle in handles:
                handle.close()
        if all(code == 0 for code in codes):
            return
        status(args.output, "infer", "retrying", exit_codes=codes, attempt=attempt)
        if attempt < 3:
            time.sleep(20)
    raise RuntimeError("Inference failed three times; see runner_infer_*.log")


def install_thesis_figures(args):
    """Copy completed figures into the prepared thesis section, then compile."""
    report = json.loads((args.output / "analysis/results/figure_report.json").read_text())
    if report["status"] != "complete":
        raise ValueError("Figure report is not complete")
    destination = args.thesis / "img_gene/vitl30_w4096"
    destination.mkdir(parents=True, exist_ok=True)
    for filename in report["figures"]:
        source = args.output / "figures" / filename
        shutil.copy2(source, destination / filename)
        if source.with_suffix(".png").is_file():
            shutil.copy2(source.with_suffix(".png"), destination / source.with_suffix(".png").name)
    run_stage(args.output, "thesis_compile", ["latexmk", "-cd", "-xelatex",
              "-interaction=nonstopmode", "-halt-on-error",
              "-outdir=/tmp/hkust-thesis-vitl30", str(args.thesis / "000_thesis.tex")], attempts=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--devices", nargs="+", default=["cuda:2", "cuda:3"])
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--thesis", type=Path)
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60 or len(set(args.devices)) != len(args.devices):
        parser.error("Use distinct devices and a poll interval of 1 to 60 seconds")
    args.output = args.output.resolve()
    lock = (args.output / "analysis_runner.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another analysis runner already owns this output directory")
    try:
        report = json.loads((args.output / "training_report.json").read_text())
        if report["status"] != "complete" or not (args.output / "model_best.pt").is_file():
            raise ValueError("Complete SAE training before starting this analysis-only runner")
        for city in names(args.output):
            if not (args.output / "cache_manifests" / f"{city.replace('/', '__')}.parquet").is_file():
                raise FileNotFoundError(f"Cache-to-image mapping unavailable: {city}")
        if not (args.output / "analysis/decoder_groups.json").is_file():
            run_stage(args.output, "cluster", command(ANALYSIS, args.output, "cluster"))
        resume_inference(args)
        run_stage(args.output, "merge", command(ANALYSIS, args.output, "merge-examples"))
        run_stage(args.output, "interpret", command(ANALYSIS, args.output, "interpret",
                                                  "--device", args.devices[-1]))
        run_stage(args.output, "figures", command(FIGURES, args.output,
                                                  "--figures", str(args.output / "figures")))
        if args.thesis:
            install_thesis_figures(args)
        status(args.output, "complete", "complete", **inference_progress(args.output))
    except Exception as exc:
        status(args.output, "failed", "failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
