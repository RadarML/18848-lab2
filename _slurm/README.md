# Slurm Job Submission

Every experiment in this repository is really two jobs: a training run (`train.py`), followed by an evaluation pass over the test set (`evaluate.py`) once that training run has produced a checkpoint. On a cluster, that means writing two `sbatch` scripts, keeping their resource requests and paths in sync, remembering to make the second one depend on the first, and doing it all again for every split and every variant you want to try.

`submit.py` does this for you. You give it a name, a version, and any hydra overrides; it renders both job scripts from the templates in `templates/`, runs a quick instantiation check to catch config errors while you are still at the terminal, and submits the pair to slurm with the evaluation job chained behind the training job.

Nothing here is required &mdash; you are welcome to write your own `train.sh` and `test.sh` and submit them with `sbatch` by hand. This is just the version that already works.

## Usage

```sh
uv run _slurm/submit.py <name> <version> --args "<hydra override>" ...
```

The `name` and `version` identify the experiment, and together determine where everything is written: `results/<name>/<version>/`. Everything after `--args` is passed through to `train.py` as a hydra override.

For example, to train the baseline on the 10% split:

```sh
uv run _slurm/submit.py baseline p10 --args "+split=p10"
```

This creates `results/baseline/p10/`, writes `train.sh` and `test.sh` into it, and submits both. When they finish, that directory holds the training logs, the checkpoints, the exported weights, and an `eval/` directory with per-trace metrics.

A few common variations:

```sh
# multiple overrides: pass each as its own quoted argument
uv run _slurm/submit.py conv p10 --args "model/tokenizer=conv" "+split=p10"

# request different resources than the environment default
uv run _slurm/submit.py baseline p100 --env.gres=gpu:h100:2 --args "+split=p100"

# check the config and render the scripts, but do not submit anything
uv run _slurm/submit.py baseline p10 --dry-run inst --args "+split=p10"

# training only, no evaluation job
uv run _slurm/submit.py baseline p10 --no-eval --args "+split=p10"
```

Run `uv run _slurm/submit.py --help` for the full list of options.

## Configuration

### Cluster defaults

Resource requests live in `environments.yaml`, keyed by environment name. The environment is detected from your hostname &mdash; anything ending in `bridges2.psc.edu` is `psc-18848`, everything else is `wave` &mdash; and can be overridden with `--env.name`.

| Environment | Partition | Account | Training GPUs | Wall limit |
| --- | --- | --- | --- | --- |
| `psc-18848` | `GPU-shared` | `ele260011p`, the 18-848 class allocation | `gpu:h100:2` | 12h |
| `psc-robo` | `ROBO` | your default slurm account | `gpu:h100:1` | 48h |
| `wave` | `batch` | your default slurm account | `gpu:1` | unset |

`psc-18848` is the default on Bridges-2, and is the only environment which sets `--account`; the others leave it unset so slurm bills your default account.

Each entry also sets `cpus` (per GPU), `mem` (per GPU), `gres_eval` (GPUs for the evaluation job &mdash; `evaluate.py` only supports one), `eval_workers` (dataloader workers during evaluation), and `nice` (lower priority for large sweeps, if you want to be polite to other users). Any field can be overridden for a single submission with the matching `--env.*` flag, e.g. `--env.queue GPU` or `--env.max-time 4:00:00`.

Each environment must also have a matching hydra config in `config/environment/`, which the job scripts compose via `+environment=<name>`; that is what sets the distributed strategy and the dataloader worker count for training.

### Paths

By default the dataset is read from `./data` and results are written to `./results`. Override with `--path.dataset` and `--path.results` if either lives somewhere else.

### Preflight checks

Before submitting anything, `submit.py` runs `train.py meta.dry_run=inst`, which composes and instantiates the exact config the job would use &mdash; on CPU, in a scratch directory, without touching the dataset. If that fails, nothing is submitted. This catches the overwhelming majority of "job died thirty seconds in" failures while you can still do something about them.

You can also ask for a dry run explicitly:

- `--dry-run inst`: run the instantiation check and render the job scripts, but do not submit.
- `--dry-run full`: additionally run a `fast_dev_run` (a single train and validation batch, on a real GPU) before rendering.
- `--dry-run skip`: submit without any checks.

`submit.py` exits `0` on success, `1` if a job could not be submitted, `2` if the instantiation check failed, and `3` if the `fast_dev_run` check failed.

## How it works

1. **Resolve the environment.** Any `--env.*` value you passed wins; anything left unset is filled in from the `environments.yaml` entry for the detected (or requested) environment.

2. **Run the preflight check.** `train.py` is invoked with `meta.dry_run=inst`, your overrides, and `+environment=<name>`, writing to a temporary directory. A non-zero exit here aborts the submission.

3. **Create the results directory.** `results/<name>/<version>/` is created up front, so slurm has somewhere to write its logs.

4. **Render the job scripts.** `templates/train.sh` and `templates/test.sh` are filled in with your name, version, paths, and resolved resources, and written to `results/<name>/<version>/train.sh` and `test.sh`. These are ordinary shell scripts &mdash; read them, and resubmit them by hand with `sbatch` if you ever want to.

    The training script runs:

    ```sh
    JAXTYPING_DISABLE=1 COLUMNS=120 uv run train.py <your overrides> \
        meta.dataset=... meta.results=... meta.name=... meta.version=... \
        meta.compile=true +environment=<name>
    ```

    `JAXTYPING_DISABLE=1` is needed because jaxtyping's runtime checks do not survive `torch.compile`, which `meta.compile=true` turns on. If training succeeds, the script then runs `uv run nrdk export`, which pulls the best checkpoint out of `checkpoints/` and writes `weights.pth` and `model.yaml` next to it.

    The evaluation script runs `evaluate.py` against the same results directory, with `--batch=8` and the environment's `eval_workers`.

5. **Submit.** `train.sh` goes to slurm first. If evaluation is enabled (the default), `test.sh` is then submitted with `--dependency afterok:<training job id>`, so it starts only if training exits cleanly. If training fails, the evaluation job is never released and slurm eventually cancels it.

6. **Write-protect the results.** Once each job finishes, it `chmod`s its outputs to read-only. This is deliberate: it stops you from accidentally overwriting a finished experiment that you have already written about. Combined with the overwrite guard in `train.py`, re-running an experiment under a name and version that already has checkpoints will abort rather than clobber them.

    If you genuinely want to redo a run &mdash; it crashed, or you changed the code &mdash; either pick a new version, or make the directory writable again and delete it:

    ```sh
    chmod -R u+w results/<name>/<version>
    rm -rf results/<name>/<version>
    ```

    Pass `--no-write-protect` if you would rather results stayed writable.

## Where things end up

```
results/<name>/<version>/
├── train.sh                  # the rendered job scripts
├── test.sh
├── train.<node>.<jobid>.log  # stdout and stderr from each job
├── train.<node>.<jobid>.err
├── test.<node>.<jobid>.log
├── test.<node>.<jobid>.err
├── .hydra/                   # the fully composed config this run used
├── events.out.tfevents.*     # tensorboard logs
├── checkpoints/              # model checkpoints
├── checkpoints.yaml          # which checkpoint was best
├── weights.pth               # exported weights, from `nrdk export`
├── model.yaml                # exported model config
└── eval/<trace>/metrics.npz  # per-sample test metrics, one file per trace
```
