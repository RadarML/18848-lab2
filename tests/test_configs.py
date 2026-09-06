"""Verify every config in config/ composes and instantiates.

For each hydra config group under config/ (e.g. model, objective,
model/encoder), every yaml file in that group is composed as an override on
default.yaml, and the resulting pipeline (transforms, datamodule,
lightningmodule, trainer) is instantiated exactly as train.py does.

This never touches the filesystem and never needs a GPU: dataset construction
is a lazy `_partial_` closure, and trace path expansion is pure string joining.
It is the same check `_slurm/submit.py` runs (as `train.py meta.dry_run=inst`)
before submitting a job, just swept across every config instead of one.
"""

import os
from pathlib import Path

import hydra
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).parent.parent
CONFIG_DIR = REPO_ROOT / "config"

# Instantiating `cfg.trainer` builds a real `lightning.Trainer`, which probes
# for a SLURM environment and warns if `srun` is installed but unused -- an
# artifact of running on a SLURM login/dev node, not something under test.
pytestmark = pytest.mark.filterwarnings(
    "ignore:The `srun` command is available"
    ":lightning.fabric.utilities.warnings.PossibleUserWarning")

# `base/` is only reachable via `+base=...` and loads a real pretrained
# checkpoint (train.py's `_load_weights`) -- not meant to stand alone.
SKIP_GROUPS = {"base"}

# Groups default.yaml selects as a list (`group: [a, b]`) rather than a
# single value (`group: a`).
LIST_STYLE_GROUPS = {"sensors", "transforms"}

# Groups not present in default.yaml's defaults list at all -- only reachable
# by *adding* a new default entry (`+group=value`), not by overriding an
# existing one (`group=value`).
OPTIONAL_GROUPS = {"environment"}

# The `radar` transform produces the `spectrum` every other transform reads,
# so it stays composed when sweeping the rest of the group.
TRANSFORMS_BASE = "radar"


def _discover_groups() -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for path in sorted(CONFIG_DIR.rglob("*.yaml")):
        group = str(path.parent.relative_to(CONFIG_DIR)).replace(os.sep, "/")
        if group == "." or group in SKIP_GROUPS:
            continue
        groups.setdefault(group, []).append(path.stem)
    return groups


def _override_for(group: str, value: str) -> str:
    prefix = "+" if group in OPTIONAL_GROUPS else ""
    if group == "transforms" and value != TRANSFORMS_BASE:
        return f"{prefix}{group}=[{TRANSFORMS_BASE},{value}]"
    if group in LIST_STYLE_GROUPS:
        return f"{prefix}{group}=[{value}]"
    return f"{prefix}{group}={value}"


def _cases() -> list:
    return [
        pytest.param(group, value, id=f"{group}={value}")
        for group, values in sorted(_discover_groups().items())
        for value in sorted(values)
    ]


def compose_cfg(overrides: list[str]) -> DictConfig:
    """Compose `default.yaml` with `overrides` applied.

    `size=pico` keeps instantiation cheap unless the case under test is
    itself a `size` config.
    """
    sized = any(o.startswith("size=") for o in overrides)
    extra = [] if sized else ["size=pico"]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name="default", overrides=extra + overrides)


def instantiate_all(cfg: DictConfig) -> None:
    """Instantiate the full pipeline, exactly as `train.py` does."""
    transforms = hydra.utils.instantiate(cfg.transforms, _convert_="all")
    hydra.utils.instantiate(
        cfg.datamodule, _convert_="all", transforms=transforms)
    hydra.utils.instantiate(
        cfg.lightningmodule, _convert_="all", transforms=transforms)
    hydra.utils.instantiate(cfg.trainer, _convert_="all")


@pytest.mark.parametrize("group,value", _cases())
def test_config_instantiates(group: str, value: str) -> None:
    """Every config in every group instantiates on top of `default.yaml`."""
    instantiate_all(compose_cfg([_override_for(group, value)]))


def _all_yaml_files() -> list[Path]:
    return sorted(CONFIG_DIR.rglob("*.yaml"))


@pytest.mark.parametrize(
    "path", _all_yaml_files(),
    ids=[str(p.relative_to(CONFIG_DIR)) for p in _all_yaml_files()])
def test_yaml_parses(path: Path) -> None:
    """Every yaml file under config/ is syntactically valid."""
    OmegaConf.load(path)


def test_default_config_instantiates() -> None:
    """The out-of-the-box config -- what a student gets with no overrides."""
    instantiate_all(compose_cfg([]))
