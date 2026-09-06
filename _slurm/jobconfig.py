"""Configuration dataclasses for SLURM job submission."""

import os
import socket
from dataclasses import dataclass

import yaml


@dataclass
class PathConfig:
    """Configuration for data and results paths.

    Attributes:
        dataset: Path to dataset directory.
        results: Path to results directory.
    """

    dataset: str = "./data"
    results: str = "./results"


@dataclass
class EnvironmentConfig:
    """Configuration for SLURM environment settings.

    Attributes:
        name: Environment name; if not specified, detected from the local
            hostname (`psc` if it ends with `bridges2.psc.edu`, else
            `wave`). Other unspecified fields load their defaults from the
            corresponding entry in `environments.yaml`.
        queue: SLURM queue/partition to submit jobs to.
        gres: Generic resources (e.g., GPUs) for training jobs.
        gres_eval: Generic resources for evaluation jobs.
        cpus: Number of CPU cores to allocate per GPU.
        mem: Amount of memory to allocate per GPU.
        max_time: Maximum wall time for each job.
        eval_workers: Number of workers for data loading during evaluation.
        nice: SLURM nice value for job priority; higher values lower
            priority.
    """

    name: str | None = None
    queue: str | None = None
    gres: str | None = None
    gres_eval: str | None = None
    cpus: str | None = None
    mem: str | None = None
    max_time: str | None = None
    eval_workers: str | None = None
    nice: int | None = None

    def set_defaults(self) -> None:
        """Set default values from environments.yaml if not specified."""
        if self.name is None:
            if socket.gethostname().endswith("bridges2.psc.edu"):
                self.name = "psc"
            else:
                self.name = "wave"

        path = os.path.join(os.path.dirname(__file__), "environments.yaml")
        with open(path) as f:
            envs = yaml.safe_load(f)
        defaults = envs.get(self.name, {})

        if self.gres is None:
            self.gres = defaults.get("gres", "gpu:1")
        if self.gres_eval is None:
            self.gres_eval = defaults.get("gres_eval", "gpu:1")
        if self.queue is None:
            self.queue = defaults.get("queue", "batch")
        if self.cpus is None:
            self.cpus = defaults.get("cpus", None)
        if self.mem is None:
            self.mem = defaults.get("mem", None)
        if self.max_time is None:
            self.max_time = defaults.get("max_time", None)
        if self.eval_workers is None:
            self.eval_workers = defaults.get("eval_workers", "16")
        if self.nice is None:
            self.nice = defaults.get("nice", None)
