# Archived Road MRLU Pipeline

This directory contains the retired road-based Minimum Road Landscape Unit
(MRLU) pipeline. It is kept for reproducibility and reference and is no longer
part of the active project README.

Contents:

- `run_experiment.py`: end-to-end subprocess orchestrator;
- `scripts/pipeline/`: the six processing stages and stage runners.
- `jobs/`: historical Slurm launch scripts that invoke the pipeline.

The archived modules still import shared helpers from `scripts/core/`. The
repository keeps a compatibility symlink at `scripts/pipeline` so older tests
and analysis scripts can import the archived modules without changing their
historical module paths. New development should use the active pipelines
documented in the repository README instead.
