# slurm

Slurm job submission utilities.

Submit a training job (and a dependent evaluation job) with:

```sh
uv run _slurm/submit.py <name> <version> --args "<hydra override>" ...
```

For example:

```sh
uv run _slurm/submit.py baseline v0 --args "size=small" "objective=lidar3d"
```

Before submitting anything, `submit.py` runs `train.py meta.dry_run=inst`,
which composes and instantiates the exact config the job would use -- on CPU,
without touching the dataset. If that fails, nothing is submitted. This
catches the overwhelming majority of "job died 30 seconds in" failures while
you are still at the terminal.

- `--dry-run inst`: run the instantiation check and render the job scripts,
  but do not submit.
- `--dry-run full`: additionally run a `fast_dev_run` (a single train/val
  batch on a real GPU) before rendering.
- `--dry-run skip`: submit without any checks.

Cluster defaults (queue, GPU type, CPUs, memory, wall time) live in
`environments.yaml`, keyed by environment name. The environment is detected
from the hostname -- `psc` on Bridges-2, otherwise `wave` -- and can be
overridden with `--env.name`. Each entry here has a matching Hydra config in
`config/environment/`, which the job script composes via `+environment=<name>`
to set the distributed strategy and dataloader worker count.
