# 18-848 Lab 2

Radar perception with Generalizable Radar Transformers (GRT).

## Setup

This repository uses [uv](https://docs.astral.sh/uv/). Install it, then:

```sh
git clone https://github.com/RadarML/18-848-lab2.git
cd 18-848-lab2
uv sync
```

That is the whole setup. There are no submodules and no `git submodule update`
step: the two RadarML libraries this lab builds on, `nrdk` and `roverd`, are
ordinary pinned dependencies fetched over HTTPS from public repositories. You
do not need an SSH key or a GitHub login to install them, and `uv sync` works
the same whether you cloned over HTTPS or SSH.

`uv.lock` pins the exact commit of every dependency, so everyone in the class
gets an identical environment. Do not delete it.

Point the lab at the dataset with `meta.dataset` (default: `./data`); a
symlink works well:

```sh
ln -s /path/to/dataset data
```

## Layout

| Path | What it is |
| --- | --- |
| `config/` | Hydra configs. This is the main thing you will edit. |
| `grt/` | GRT building blocks (`SpectrumTokenizer`, ...) referenced by `config/model/`. |
| `train.py` | Training entry point. |
| `evaluate.py` | Evaluation entry point; writes per-trace metrics and renders. |
| `_slurm/` | Cluster job submission. See [`_slurm/README.md`](_slurm/README.md). |
| `tests/` | Config sweep -- checks every config still composes and instantiates. |

The model is assembled by Hydra from four config groups, so most experiments
are a matter of composing existing pieces rather than writing code:

- `model/tokenizer` -- turns the radar spectrum into tokens
- `model/encoder` -- a vanilla `torch.nn.TransformerEncoder`
- `model/decoder` -- queries the encoded tokens at output positions
- `objective` -- the loss and its visualizations

## Running

Every run needs a `meta.name` and `meta.version`; results are written to
`${meta.results}/${meta.name}/${meta.version}`, and an existing directory is
never silently overwritten.

```sh
# Check that a config composes and instantiates -- CPU only, no dataset,
# a few seconds. Do this before every real run.
uv run train.py meta.name=baseline meta.version=v0 meta.dry_run=inst

# A tiny end-to-end run, to confirm data loading and the training loop work.
uv run train.py +environment=debug

# A real run.
uv run train.py meta.name=baseline meta.version=v0 size=small

# Evaluate the result.
uv run evaluate.py results/baseline/v0
```

`meta.dry_run=full` goes one step further and runs a single train/val batch
(lightning's `fast_dev_run`). It prints this warning, which is expected and
harmless:

```
WARNING  NRDKLightningModule: Tried to log visualizations, but the logger
         does not implement the `LoggerWithImages` interface.
```

Under `fast_dev_run`, lightning replaces the configured logger with a
`DummyLogger`, which has no image support -- so the visualization hook has
nowhere to write and says so. It does not mean anything is wrong with your
config. A real run uses the TensorBoard logger and logs visualizations
normally.

Configs are composed with Hydra overrides, e.g.:

```sh
uv run train.py meta.name=big meta.version=v0 \
    size=medium objective=lidar2d model/decoder=lidar2d
```

Watch training with `uv run tensorboard --logdir results` (or `--logdir
debug_results` for `+environment=debug` runs).

## Cluster

```sh
uv run _slurm/submit.py baseline v0 --args "size=small"
```

This runs the `meta.dry_run=inst` check first and refuses to submit if it
fails, then queues a training job plus a dependent evaluation job. See
[`_slurm/README.md`](_slurm/README.md).

## Tests

```sh
uv run pytest tests/
```

`tests/test_configs.py` composes every config in `config/` on top of
`default.yaml` and instantiates the full pipeline. If you add a config, this
will tell you it is broken before you spend a GPU allocation finding out.

Lint and type-check with `uv run ruff check .` and `uv run pyright`.

## Pre-commit

Ruff, pyright, and pytest also run automatically on every commit. Enable them
once per clone:

```sh
uv run pre-commit install
```

To run them by hand over the whole repo:

```sh
uv run pre-commit run --all-files
```

Note that pre-commit only sees files git already tracks, so a brand-new file
is skipped until you `git add` it.

## Dependencies

Notes for whoever maintains this repo between semesters. Students should not
need any of this.

**`uv.lock` must stay committed.** It records the exact commit of every
dependency and is the only thing that makes the class environment
reproducible.

**`nrdk` and `roverd` are not on PyPI**, and come from public GitHub repos over
HTTPS. Because they are public, this needs no SSH key and no credentials, which
is why they are dependencies rather than submodules.

**`roverd` cannot be pinned to a tag, and is deliberately not listed in
`[tool.uv.sources]`.** It arrives transitively via `nrdk[roverd]`, and uv
honors nrdk's own source declaration for it -- which is untagged, i.e. it
follows red-rover's default branch. Adding a conflicting `roverd` source here
does not override that: uv unifies the two git requirements only when they
resolve to the same commit, and otherwise fails with `conflicting URLs for
package roverd`. So pinning roverd to anything other than red-rover's current
HEAD will not resolve at all.

What pins roverd in practice is `uv.lock`. The `roverd[video] == 0.3.5`
version constraint is a tripwire: if red-rover's default branch moves to a new
version, `uv lock` fails loudly instead of silently upgrading students. The
real fix is upstream -- pin the `roverd` source in nrdk's own `pyproject.toml`
to a tag.

**`pyg_lib` is tied to the `torch` pin.** It is the compiled accelerator behind
`torch_geometric.nn.pool.knn`, which the point-cloud metrics require whenever
an objective sets `require_knn: True` (the default `objective=lidar3d` does).
Its wheels are not on PyPI and are specific to a (torch, CUDA, Python) triple,
so they come from the `find-links` index in `[tool.uv]`, currently
torch 2.12.0 + cu130 + cp312. **Bumping torch or CUDA means bumping that URL
too.** That index ships `manylinux_x86_64` and `win_amd64` wheels only -- there
is no macOS or ARM build, so `uv sync` will fail on an Apple Silicon laptop.

To bump `nrdk`: change the tag in `[tool.uv.sources]`, run `uv lock`, run
`uv run pytest tests/`, and commit `uv.lock`.

## Reference

Documentation for the underlying libraries:

- [`nrdk`](https://radarml.github.io/nrdk/) -- models, objectives, training framework
- [`red-rover`](https://radarml.github.io/red-rover/) -- `roverd` dataset format
- [`xwr`](https://radarml.github.io/xwr/) -- TI mmWave radar processing
- [`abstract-dataloader`](https://radarml.github.io/abstract-dataloader/) -- transform/pipeline API
