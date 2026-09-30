# Feature-MAE Scale-up Training Status

Snapshot time: `2026-09-25T00:59:17Z`

This document records the recoverable state of the 21-task Feature-MAE scale-up
suite after a server interruption. It is an operational checkpoint, not a final
scientific result.

## Recoverable state

- Completed tasks: `6/21`.
- Interrupted task: `scale_n10000_w0128_s42` (`7/21`).
- Latest valid checkpoint: epoch `9/10`, optimizer step `5,625`.
- Latest train loss: `0.132354`.
- Latest validation loss: `0.132091`.
- Current checkpoint size: `32,837,660` bytes.
- Remaining tasks: `14` planned tasks plus the final epoch of task 7.
- At snapshot time, no runner, trainer, or watcher process was alive. The
  persisted `runner_state.json` still said `running`, so process presence must
  be checked rather than trusting that file alone.

The checkpoint was loaded successfully on CPU and contains the model,
optimizer, scheduler, epoch, optimizer-step counter, best validation loss, and
complete history required for an exact epoch-boundary resume.

## Completed runs

| Experiment | Images/city | Width | Seed | Best epoch | Train loss | Validation loss | Peak GPU (GiB) | Elapsed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `scale_n10000_w0512_s42` | 10,000 | 512 | 42 | 10 | 0.109750 | 0.109751 | 3.215 | 3:30:31 |
| `scale_n00500_w0512_s42` | 500 | 512 | 42 | 10 | 0.109095 | 0.110007 | 3.215 | 0:47:38 |
| `scale_n01000_w0512_s42` | 1,000 | 512 | 42 | 10 | 0.109541 | 0.109910 | 3.215 | 0:47:43 |
| `scale_n02000_w0512_s42` | 2,000 | 512 | 42 | 10 | 0.109763 | 0.109899 | 3.215 | 0:48:39 |
| `scale_n04000_w0512_s42` | 4,000 | 512 | 42 | 10 | 0.109814 | 0.109816 | 3.215 | 0:51:48 |
| `scale_n08000_w0512_s42` | 8,000 | 512 | 42 | 10 | 0.109818 | 0.109796 | 3.215 | 2:44:07 |

These values are copied from each run's `training_report.json`. The model files
and reports under `outputs/experiments/` remain local and are intentionally not
committed.

## Safety and recovery changes

The training loop now supports a one-batch CPU prefetch queue. It overlaps disk
reads with GPU work without changing sample order, effective batch size, model,
or optimizer steps. Extra host memory is bounded to one training batch.

The runner no longer pins training to a restricted CPU set. A separate watcher
records every 30 seconds:

- system CPU utilization, load, package temperature, and process CPU/RSS;
- GPU utilization, memory utilization, used/free memory, temperature, and power;
- available RAM, Swap, and disk capacity;
- runner/trainer PIDs and current experiment;
- checkpoint, best-model, log, and report presence, size, and modification age.

The watcher stops the trainer after two consecutive critical samples when any
of the following limits is crossed:

| Resource | Safety limit |
|---|---:|
| Available RAM | `< 4 GiB` |
| GPU temperature | `>= 84 C` |
| CPU package temperature | `>= 90 C` |
| Free GPU memory | `< 512 MiB` |
| Free experiment-disk space | `< 500 GiB` |

Stopping the trainer preserves the last atomic epoch checkpoint. CPU usage is
observed but not affinity-limited, as requested.

## Resume procedure

From the repository root:

```bash
python3 -m scripts.multicity.run_feature_mae_scaleup
```

In a second persistent session:

```bash
python3 -m scripts.multicity.watch_feature_mae_scaleup --interval 30
```

The runner skips the six runs that already contain valid
`training_report.json` files and resumes `scale_n10000_w0128_s42` at epoch 10.
It must not delete or overwrite the existing run directory.

## Files intentionally excluded from GitHub

- `checkpoint_last.pt` and `model_best.pt`;
- DINOv3 token arrays and image data;
- run logs, caches, and the live registry under `outputs/experiments/`;
- all other generated artifacts covered by `.gitignore`.

Only source code and this small text snapshot are intended for the pull request.
