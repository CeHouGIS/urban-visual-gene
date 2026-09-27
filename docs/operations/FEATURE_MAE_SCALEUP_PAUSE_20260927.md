# Feature-MAE 21-run scale-up pause archive

## Pause state

- Paused by user at: `2026-09-27T09:55:05Z`
- Queue progress: **11 completed / 21 total**
- Interrupted item: **12/21**, `scale_n10000_w0512_s43`
- Interrupted item progress: **2/10 epochs completed**
- Remaining after the interrupted item: **9 planned runs**
- Runner PID `6763`, watcher PID `6766`, and trainer PID `9543` were terminated normally with `SIGTERM`.
- Post-pause GPU state: `90 MiB` used, `0%` utilization, `40 C`.
- No scale-up runner, watcher, or trainer process remained after the pause.

The live registry was changed from `running` to `paused`, and
`runner_state.json` was changed to `paused_by_user`. No completed model or
checkpoint was deleted.

## Resume checkpoint

The interrupted run has a complete epoch-boundary checkpoint:

```text
/workplace/urban_visual_gene/outputs/experiments/dinov3_multicity/feature_mae_scaleup/runs/scale_n10000_w0512_s43/checkpoint_last.pt
```

- Size: `179,668,444 bytes`
- SHA-256: `eb7b1e7bc1a637883746a0a0bde9d93cbc62902fca30c2a8e6c7b94fb0e3c53c`
- Checkpoint modification time: `2026-09-27T09:50:28Z`
- Resume point: epoch 3 of 10
- Epoch 1: train loss `0.338233`, validation loss `0.213950`
- Epoch 2: train loss `0.192178`, validation loss `0.174778`

`model_best.pt` also exists in the same directory. `training_report.json` does
not exist yet because this run is incomplete.

## Authoritative 21-run registry snapshot

| # | Experiment | Group | Status at pause | Best validation loss |
|---:|---|---|---|---:|
| 1 | `scale_n10000_w0512_s42` | reference | completed | 0.109751 |
| 2 | `scale_n00500_w0512_s42` | sample curve | completed | 0.110007 |
| 3 | `scale_n01000_w0512_s42` | sample curve | completed | 0.109910 |
| 4 | `scale_n02000_w0512_s42` | sample curve | completed | 0.109899 |
| 5 | `scale_n04000_w0512_s42` | sample curve | completed | 0.109816 |
| 6 | `scale_n08000_w0512_s42` | sample curve | completed | 0.109796 |
| 7 | `scale_n10000_w0128_s42` | width curve | completed | 0.131856 |
| 8 | `scale_n10000_w0256_s42` | width curve | completed | 0.123335 |
| 9 | `scale_n10000_w1024_s42` | width curve | completed | 0.101784 |
| 10 | `scale_n01000_w0512_s43` | sample replicate | completed | 0.112147 |
| 11 | `scale_n04000_w0512_s43` | sample replicate | completed | 0.112118 |
| 12 | `scale_n10000_w0512_s43` | sample replicate | **paused after epoch 2** | 0.174778 so far |
| 13 | `scale_n10000_w0128_s43` | width replicate | planned | — |
| 14 | `scale_n10000_w0256_s43` | width replicate | planned | — |
| 15 | `scale_n10000_w1024_s43` | width replicate | planned | — |
| 16 | `scale_n01000_w0512_s44` | sample replicate | planned | — |
| 17 | `scale_n04000_w0512_s44` | sample replicate | planned | — |
| 18 | `scale_n10000_w0512_s44` | sample replicate | planned | — |
| 19 | `scale_n10000_w0128_s44` | width replicate | planned | — |
| 20 | `scale_n10000_w0256_s44` | width replicate | planned | — |
| 21 | `scale_n10000_w1024_s44` | width replicate | planned | — |

The earlier separate `seed44_fast` attempt at
`scale_n01000_w0512_s44` failed and is not authoritative. It remains planned
in the main 21-run registry and should be run only by the main serial runner.

## Live state and logs

```text
/workplace/urban_visual_gene/outputs/experiments/dinov3_multicity/feature_mae_scaleup/experiment_registry.csv
/workplace/urban_visual_gene/outputs/experiments/dinov3_multicity/feature_mae_scaleup/runner_state.json
/workplace/urban_visual_gene/outputs/experiments/dinov3_multicity/feature_mae_scaleup/runner.log
/workplace/urban_visual_gene/outputs/experiments/dinov3_multicity/feature_mae_scaleup/watcher.log
```

## Safe resume procedure

First update the working tree so the runner includes the zero-prefetch safety
fix. Then start only the main runner and one watcher:

```bash
cd /workplace/urban_visual_gene
git pull --ff-only origin main

taskset -c 0-3 nohup python3 -m scripts.multicity.run_feature_mae_scaleup \
  --root /workplace/urban_visual_gene/outputs/experiments/dinov3_multicity/feature_mae_scaleup \
  --prefetch-batches 0 \
  >> /workplace/urban_visual_gene/outputs/experiments/dinov3_multicity/feature_mae_scaleup/runner_stdout.log 2>&1 &

taskset -c 0-3 nohup python3 -m scripts.multicity.watch_feature_mae_scaleup \
  --root /workplace/urban_visual_gene/outputs/experiments/dinov3_multicity/feature_mae_scaleup \
  --interval 30 \
  --ram-min-gib 8 \
  --gpu-temp-max 82 \
  --cpu-temp-max 88 \
  --gpu-free-min-mib 1024 \
  --disk-free-min-gib 700 \
  --consecutive-critical 2 \
  >> /workplace/urban_visual_gene/outputs/experiments/dinov3_multicity/feature_mae_scaleup/watcher_stdout.log 2>&1 &
```

On restart, the runner will skip the 11 runs with existing
`training_report.json` files. Run 12 will load `checkpoint_last.pt` and report
`resuming at epoch 3`; it must not restart from epoch 1. CPU affinity must
continue to exclude logical CPUs 8 and 9.

Do not start the old `seed44_fast` runner in parallel with the main queue.
