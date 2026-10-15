"""Build the image path list for the 307-city formal sample only."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import formal.gpu_run as run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city-map", required=True)
    ap.add_argument("--baidu-map", required=True)
    ap.add_argument("--sample-json", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    google = json.loads(Path(args.city_map).read_text())
    baidu = json.loads(Path(args.baidu_map).read_text())
    sample = json.loads(Path(args.sample_json).read_text())
    run.CITY_DIR.clear(); run.CITY_DIR.update(google)
    run.META_CITY_DIR.clear(); run.META_CITY_DIR.update(google)
    run.BAIDU_CITY_DIR.clear(); run.BAIDU_CITY_DIR.update(baidu)
    run.CITY_DIR.update(baidu); run.META_CITY_DIR.update(baidu)

    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"google": 0, "baidu": 0, "missing": 0}
    cities = list(sample.items())
    if not 0 <= args.shard < args.nshards:
        ap.error("shard must be in [0, nshards)")
    cities = [(index, city, quota) for index, (city, quota) in enumerate(cities)
              if index % args.nshards == args.shard]
    with output.open("w", encoding="utf-8") as out:
        for local_index, (index, city, quota) in enumerate(cities, 1):
            if city in baidu:
                panos = run.disk_panos(city, int(quota), random.Random(0))
                for pano in panos:
                    path = run.imgpath(city, pano, 0)
                    if path.is_file():
                        out.write(f"{city}\t{path}\n"); counts["baidu"] += 1
                    else:
                        counts["missing"] += 1
                continue

            candidates = run.stratified_panos(city, int(quota), seed=0)
            selected = []
            for pano in candidates:
                if all(run.imgpath(city, pano, heading).is_file() for heading in run.HEADINGS):
                    selected.append(pano)
                if len(selected) >= int(quota):
                    break
            for pano in selected:
                for heading in run.HEADINGS:
                    path = run.imgpath(city, pano, heading)
                    out.write(f"{city}\t{path}\n"); counts["google"] += 1
            counts["missing"] += max(0, int(quota) - len(selected)) * len(run.HEADINGS)
            if local_index % 5 == 0 or local_index == len(cities):
                print(f"[shard {args.shard}] {city}: {len(selected)} panos ({local_index}/{len(cities)})", flush=True)
    print(json.dumps(counts, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
