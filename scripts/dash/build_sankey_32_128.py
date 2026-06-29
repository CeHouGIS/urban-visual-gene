#!/usr/bin/env python3
"""Sankey data: how the 32-basis dominant basis splits into the 128-basis one.

Pools all road nodes across the 4 cities (32-basis dominant from <city>_attr.bin
byte 0; 128-basis dominant from <city>_dom128.bin) and cross-tabulates them.
Writes dashboard/data/sankey_32_128.json:
  {nodes:[{name,label,itemStyle:{color}}], links:[{source,target,value}], total, n_links}
Node colours match the map palettes (32-HSL wheel / 128 golden-angle).
  OMP_NUM_THREADS=1 python -m scripts.dash.build_sankey_32_128
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "dashboard", "data")
CITIES = ["HongKong", "Singapore", "Amsterdam", "CapeTown"]
K32, K128 = 32, 128


def hsl2rgb(h, s, l):
    def f(n):
        k = (n + h * 12) % 12
        a = s * min(l, 1 - l)
        return round(255 * (l - a * max(-1, min(k - 3, 9 - k, 1))))
    return f(0), f(8), f(4)


def hexc(rgb):
    return "#%02x%02x%02x" % rgb


def pal32(k):
    return hexc(hsl2rgb((k * 360 / 32) / 360, 0.62, 0.58))


def pal128(j):
    h = (j * 137.508) % 360
    l = 0.50 if j % 2 else 0.64
    return hexc(hsl2rgb(h / 360, 0.66, l))


def main():
    M = np.zeros((K32, K128), np.int64)
    for c in CITIES:
        attr = np.fromfile(os.path.join(OUT, f"{c}_attr.bin"), np.uint8).reshape(-1, 4)
        dom32 = attr[:, 0].astype(np.int64)
        dom128 = np.fromfile(os.path.join(OUT, f"{c}_dom128.bin"), np.uint8).astype(np.int64)
        n = min(len(dom32), len(dom128))
        np.add.at(M, (dom32[:n], dom128[:n]), 1)
        print(f"[sankey] {c}: {n:,} nodes", flush=True)

    total = int(M.sum())
    rowsum = M.sum(1)            # per 32-basis node count
    # keep a link if it is >=2% of its 32-basis's nodes AND >= 800 nodes absolute,
    # so the diagram shows each coarse basis's main fine children without clutter
    nodes, idx, links = [], {}, []

    def node(name, label, color):
        if name not in idx:
            idx[name] = len(nodes)
            nodes.append({"name": name, "label": label, "itemStyle": {"color": color}})
        return idx[name]

    kept = 0
    for k in range(K32):
        if rowsum[k] == 0:
            continue
        for j in range(K128):
            v = int(M[k, j])
            if v >= 800 and v >= 0.02 * rowsum[k]:
                node(f"v32_{k}", f"32#{k}", pal32(k))
                node(f"v128_{j}", f"128#{j}", pal128(j))
                links.append({"source": f"v32_{k}", "target": f"v128_{j}", "value": v})
                kept += 1

    used32 = [int(k) for k in range(K32) if rowsum[k] > 0]
    used128 = [int(j) for j in range(K128) if M[:, j].sum() > 0]
    out = {"nodes": nodes, "links": links, "total": total, "n_links": kept,
           "n_left": sum(1 for n in nodes if n["name"].startswith("v32_")),
           "n_right": sum(1 for n in nodes if n["name"].startswith("v128_")),
           # full cross-tab (rows=32 basis, cols=128 basis) for the matrix heatmap
           "matrix": M.tolist(), "rowsum": rowsum.tolist(),
           "used32": used32, "used128": used128,
           "col32": [pal32(k) for k in range(K32)],
           "col128": [pal128(j) for j in range(K128)]}
    json.dump(out, open(os.path.join(OUT, "sankey_32_128.json"), "w"), separators=(",", ":"))
    print(f"[sankey] total={total:,} links={kept} left={out['n_left']} right={out['n_right']} "
          f"-> sankey_32_128.json", flush=True)


if __name__ == "__main__":
    main()
