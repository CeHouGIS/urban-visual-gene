"""Screen every downloaded street-view image for black frames and tunnels.

The source images are never modified.  The command writes a CSV audit table,
plus ``keep.txt`` and ``drop.txt`` path lists, so downstream samplers can use
the retained set without rescanning the images.  It is intentionally a
streaming job: paths are discovered by ``find`` and scores are written as they
arrive, keeping memory bounded even for the full NAS catalogue.

    python -m formal.quality.screen_all_images --workers 48

Use ``--limit`` for a smoke test.  A shard can be run independently with
``--shard N --nshards M``; the path hash makes shards disjoint and reproducible.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import subprocess
import time
from pathlib import Path

# Each image is independent. Prevent scipy/sklearn from starting a full BLAS
# thread pool inside every multiprocessing worker.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from .tiny_qc import QualityGate

DEFAULT_ROOTS = {
    "google": "/host/root/mnt/nas/huangyj/GoogleSV/images",
    "baidu": "/host/root/mnt/nas/liyong/baidusv_china/images",
}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_GATE: QualityGate | None = None


def _init_worker(model: str) -> None:
    global _GATE
    _GATE = QualityGate(model)
    if not _GATE.model_loaded:
        raise RuntimeError(f"quality model is unavailable: {_GATE.model_path}")


def _score(path: str):
    global _GATE
    try:
        from PIL import Image

        with Image.open(path) as image:
            result = _GATE.score_view(image.convert("RGB"))
        p_black, p_tunnel, hard_black, hard_tunnel = result
        black = bool(hard_black or p_black >= _GATE.black_threshold)
        tunnel = bool(hard_tunnel or p_tunnel >= _GATE.tunnel_threshold)
        reason = "black" if black else ("tunnel" if tunnel else "")
        return (path, p_black, p_tunnel, int(black), int(tunnel), 1, reason)
    except Exception as exc:
        return (path, 0.0, 0.0, 0, 0, 0, f"unreadable:{type(exc).__name__}")


def _roots(values: list[str] | None) -> dict[str, str]:
    roots = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"root must be NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        roots[name] = path
    return roots or DEFAULT_ROOTS.copy()


def _iter_paths(roots: dict[str, str], shard: int, nshards: int):
    for dataset, root in roots.items():
        if not Path(root).is_dir():
            print(f"[skip] missing root {dataset}: {root}", flush=True)
            continue
        command = ["find", root, "-type", "f", "("]
        for index, ext in enumerate(sorted(EXTENSIONS)):
            if index:
                command.append("-o")
            command.extend(["-iname", f"*{ext}"])
        command.extend([")", "-print0"])
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=1024 * 1024)
        try:
            assert proc.stdout is not None
            while True:
                raw = proc.stdout.read(1024 * 1024)
                if not raw:
                    break
                for item in raw.split(b"\0"):
                    if not item:
                        continue
                    path = os.fsdecode(item)
                    digest = int.from_bytes(hashlib.blake2b(path.encode(), digest_size=8).digest(), "big")
                    if digest % nshards == shard:
                        yield dataset, path
            if proc.wait() != 0:
                raise RuntimeError(f"find failed for {root}")
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()


def run(args: argparse.Namespace) -> dict:
    root_values = list(args.root or [])
    if args.city_map:
        google_root = Path(os.environ.get("GOOGLE_SV_ROOT", "/host/root/mnt/nas/huangyj/GoogleSV")) / "images"
        city_map = json.loads(Path(args.city_map).read_text())
        root_values.extend(f"google_{city}={google_root / relative}" for city, relative in city_map.items())
    if args.baidu_map:
        baidu_map = json.loads(Path(args.baidu_map).read_text())
        root_values.extend(f"baidu_{city}={path}" for city, path in baidu_map.items())
    roots = _roots(root_values)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / f"images.part{args.shard:04d}.csv"
    keep_path = output / f"keep.part{args.shard:04d}.txt"
    drop_path = output / f"drop.part{args.shard:04d}.txt"
    done_path = output / f"part{args.shard:04d}.done.json"
    if done_path.exists() and not args.force:
        print(f"[skip] {done_path} exists", flush=True)
        return json.loads(done_path.read_text())

    fields = ["dataset", "path", "p_black", "p_tunnel", "black", "tunnel", "readable", "reason"]
    mode = "ab" if args.resume and csv_path.exists() else "wb"
    # CSV is append-safe after a process restart; a partial final line is
    # discarded only when --resume is explicitly requested.
    if mode == "ab":
        with csv_path.open("rb+") as f:
            f.seek(0, os.SEEK_END)
            # Find the last newline without loading the file.
            pos = f.tell()
            while pos and f.read(1) != b"\n":
                pos -= 1
                f.seek(pos)
            f.truncate(pos + 1)
    else:
        csv_path.write_text(",".join(fields) + "\n")
    stats = {"seen": 0, "readable": 0, "keep": 0, "black": 0, "tunnel": 0, "unreadable": 0}
    started = time.monotonic()
    batch_size = min(args.batch, args.limit) if args.limit else args.batch
    if args.path_list:
        def listed_paths():
            with Path(args.path_list).open(encoding="utf-8") as source:
                for line in source:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    parts = line.split("\t", 1)
                    yield (parts[0] if len(parts) == 2 else "sample"), (parts[-1])
        path_stream = listed_paths()
    else:
        path_stream = ((dataset, path) for dataset, path in _iter_paths(roots, args.shard, args.nshards))
    path_stream = (item for item in path_stream)
    if args.limit:
        path_stream = itertools.islice(path_stream, args.limit)
    def dataset_for(path: str) -> str:
        for dataset, root in roots.items():
            root_prefix = root.rstrip(os.sep) + os.sep
            if path.startswith(root_prefix):
                return dataset
        return "unknown"

    with csv_path.open("a", newline="") as csv_file, keep_path.open("a") as keep_file, drop_path.open("a") as drop_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        pool = mp.Pool(processes=max(1, args.workers), initializer=_init_worker,
                       initargs=(args.model,))
        try:
            path_to_dataset = {}
            def just_paths():
                for dataset, path in path_stream:
                    path_to_dataset[path] = dataset
                    yield path
            for record in pool.imap_unordered(_score, just_paths(), chunksize=16):
                _write_record(writer, keep_file, drop_file, path_to_dataset.get(record[0], dataset_for(record[0])), record, stats)
        finally:
            pool.close(); pool.join()
    stats.update({"dataset_roots": roots, "shard": args.shard, "nshards": args.nshards,
                  "elapsed_s": round(time.monotonic() - started, 1)})
    done_path.write_text(json.dumps(stats, indent=2) + "\n")
    print("[complete]", json.dumps(stats, ensure_ascii=False), flush=True)
    return stats


def _write_record(writer, keep_file, drop_file, dataset, record, stats):
    path, p_black, p_tunnel, black, tunnel, readable, reason = record
    bad = bool(black or tunnel or not readable)
    row = {"dataset": dataset, "path": path, "p_black": f"{p_black:.6f}", "p_tunnel": f"{p_tunnel:.6f}",
           "black": black, "tunnel": tunnel, "readable": readable, "reason": reason}
    writer.writerow(row)
    stats["seen"] += 1
    if not readable:
        stats["unreadable"] += 1
    else:
        stats["readable"] += 1
    stats["black"] += black
    stats["tunnel"] += tunnel
    if bad:
        drop_file.write(f"{dataset}\t{reason}\t{path}\n")
    else:
        stats["keep"] += 1
        keep_file.write(f"{dataset}\t{path}\n")
    if stats["seen"] % 10000 == 0:
        keep_file.flush(); drop_file.flush()
        print(f"[{stats['seen']:,}] keep={stats['seen']-stats['black']-stats['tunnel']-stats['unreadable']:,} "
              f"black={stats['black']:,} tunnel={stats['tunnel']:,} rate={stats['seen']:.0f}/elapsed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", help="dataset=directory (repeatable)")
    parser.add_argument("--path-list", default="", help="tab-separated dataset/path list; bypass directory scanning")
    parser.add_argument("--city-map", default="", help="Google city map; scan only these country/city roots")
    parser.add_argument("--baidu-map", default="", help="Baidu city map; scan only these absolute roots")
    parser.add_argument("--output", default="/host/root/mnt/nas/huangyj/GoogleSV/urban_visual_gene/all_image_qc")
    parser.add_argument("--model", default=str(Path(__file__).with_name("qc_model.joblib")))
    parser.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 1))
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard < args.nshards:
        parser.error("shard must be in [0, nshards)")
    run(args)


if __name__ == "__main__":
    main()
