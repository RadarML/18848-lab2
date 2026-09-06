"""SLURM job submission script for GRT training and evaluation jobs."""

import logging
import os
import re
import subprocess
import tempfile
from typing import Literal

import jinja2
import tyro
from jobconfig import EnvironmentConfig, PathConfig
from rich.console import Console
from rich.logging import RichHandler

logger = logging.getLogger(__name__)


def create_job_script(
    template_file: str,
    context: dict,
    output_file: str
) -> None:
    """Create a job script from a template.

    Args:
        template_file: Path to the template file
        context: Variables to pass to the template
        output_file: Path where the rendered job script will be saved
    """
    template_dir = os.path.dirname(template_file)
    jinja = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))

    template = jinja.get_template(os.path.basename(template_file))
    content = template.render(**context)

    # Write to output file
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write(content)

    # Make executable
    os.chmod(output_file, 0o755)


def submit_slurm_job(
    job: str, dependency: str | int | None = None
) -> str | None:
    """Submit a job to SLURM and return the job ID.

    Args:
        job: Path to the job script to submit
        dependency: Optional job ID to depend on (format: "afterok:12345")

    Returns:
        Job ID if successful, `None` if failed
    """
    cmd = ['sbatch']
    if dependency:
        cmd.extend(['--dependency', f"afterok:{dependency}"])
    cmd.append(job)

    result = None
    try:
        logger.info(f"Submitting job: [cyan]{job}[/cyan]")
        if dependency:
            logger.info(f" ├─ with dependency: [yellow]{dependency}[/yellow]")

        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        logger.debug(f" ├─ [dim]{output}[/dim]")

        match = re.search(r'Submitted batch job (\d+)', output)
        if match:
            job_id = match.group(1)
            logger.info(f" └─ Job ID: [bold green]{job_id}[/bold green]")
            return job_id
        else:
            logger.warning("Could not extract job ID from SLURM output")
            return None

    except Exception as e:
        logger.error(f"Failed to submit job [red]{job}[/red]: {e}")
        if result and result.stderr:
            logger.error(f" └─ stderr: [red]{result.stderr.strip()}[/red]")
        return None


def run_dry_run(
    mode: Literal["inst", "full"], name: str, version: str,
    args: list[str], path: PathConfig, env: EnvironmentConfig
) -> bool:
    """Run `train.py meta.dry_run=<mode>` in a scratch directory.

    Mirrors the overrides `templates/train.sh` passes to the real job, so
    the check exercises the exact config that would be submitted, only
    swapping in a throwaway `meta.results`/`name`/`version`.

    Args:
        mode: dry-run mode to pass to `train.py` (`inst` or `full`).
        name: Job name identifier (for logging only).
        version: Job version identifier (for logging only).
        args: Additional arguments to pass to train script.
        path: File path configuration.
        env: Cluster environment configuration.

    Returns:
        `True` if the dry run passed (exit code 0), `False` otherwise.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        cmd = [
            "uv", "run", "train.py", *args,
            f"meta.dataset={path.dataset}",
            f"meta.results={tmp_dir}",
            "meta.name=dryrun", "meta.version=dryrun",
            f"meta.dry_run={mode}",
            f"+environment={env.name}",
        ]
        if mode == "inst":
            logger.info(
                "Verifying config instantiation for "
                f"[bold]{name}.{version}[/bold]...")
        else:
            logger.info("Running fast_dev_run...")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(
                f"Dry run (mode={mode}) failed; no jobs will be submitted.")
            logger.error(result.stdout)
            logger.error(result.stderr)
            return False
        logger.info(f"[bold green]✓[/bold green] Dry run (mode={mode}) passed")
        return True


def submit_job(
    name: str, version: str, /,
    dependency: str | None = None,
    args: list[str] = [],
    dry_run: Literal["inst", "full", "skip"] | None = None,
    eval: bool = True,
    path: PathConfig = PathConfig(),
    env: EnvironmentConfig = EnvironmentConfig(),
    write_protect: bool = True,
    verbose: int = logging.INFO
) -> int:
    """Submit a GRT train/eval job to slurm.

    Unspecified values are set to the default configuration specified in
    `environments.yaml` for the given environment.

    Before doing anything else, this runs `train.py`'s `meta.dry_run=inst`
    check (and, if `dry_run="full"`, also the `meta.dry_run=full` check) to
    validate the config instantiates correctly; submission is aborted if
    either check fails.

    Dry runs can also be manually requested:

    - `--dry-run inst`: only check instantiation without creating or submitting
        the job script.
    - `--dry-run full`: run a `fast_dev_run` check, also without creating or
        submitting.
    - `--dry-run skip`: skip all dry run checks and submit the job as-is.

    Args:
        name: Job name identifier
        version: Job version identifier
        dependency: Optional job ID to depend on
        args: Additional arguments to pass to train script
        dry_run: Dry run mode.
        eval: Whether to create and submit the evaluation job
        path: File path configuration
        env: Cluster environment configuration
        write_protect: Set results to read-only after job completion
        verbose: Logging verbosity level

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    console = Console()
    logging.basicConfig(
        level=verbose, format="%(message)s", datefmt="[%X]",
        handlers=[RichHandler(
            console=console, rich_tracebacks=True, markup=True)])

    env.set_defaults()
    logger.info(f"Creating job for [bold]{name}.{version}[/bold]")

    if dry_run == "full":
        if not run_dry_run("full", name, version, args, path, env):
            return 3
    elif dry_run != "skip":
        if not run_dry_run("inst", name, version, args, path, env):
            return 2

    # Pre-create results directory so SLURM can write logs there
    job_dir = os.path.join(path.results, name, version)
    os.makedirs(job_dir, exist_ok=True)

    # Base template variables
    common_vars = {
        "name": name,
        "version": version,
        "dataset": path.dataset,
        "results": path.results,
        "write_protect": write_protect,
        "env": env,
    }
    logging.debug(f"Job context: {common_vars}")

    # Define file paths
    train_file = os.path.join(job_dir, "train.sh")
    test_file = os.path.join(job_dir, "test.sh")

    # Create job scripts from templates
    train_vars = {**common_vars, "args": args}
    templates = os.path.join(os.path.dirname(__file__), "templates")

    create_job_script(
        os.path.join(templates, "train.sh"), train_vars, train_file)

    if eval:
        test_vars = {**common_vars, "args": ["--batch=8"]}
        create_job_script(
            os.path.join(templates, "test.sh"), test_vars, test_file)

    if dry_run in ("inst", "full"):
        logger.info("Created job scripts:")
        logger.info(f" ├─ Training: [cyan]{train_file}[/cyan]")
        if eval:
            logger.info(f" └─ Evaluation: [cyan]{test_file}[/cyan]")
        logger.info(
            f"[bold green]✓[/bold green] Dry run completed for "
            f"[bold]{name}.{version}[/bold]")
        return 0

    train_job_id = submit_slurm_job(train_file, dependency=dependency)
    if not train_job_id:
        return 1

    if eval:
        test_job_id = submit_slurm_job(test_file, dependency=train_job_id)
        if not test_job_id:
            return 1

    return 0


if __name__ == "__main__":
    tyro.cli(submit_job)
