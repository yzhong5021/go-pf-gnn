"""Monitor a Slurm job by name, resubmitting on termination."""

from __future__ import annotations

import argparse
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)

TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "PREEMPTED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
}
RESUBMIT_STATES = {
    "FAILED",
    "CANCELLED",
    "PREEMPTED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
}


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        LOGGER.warning("command failed (%s): %s", " ".join(cmd), stderr or "unknown error")
    return result.stdout.strip()


def _squeue_state(job_name: str) -> Optional[str]:
    output = _run(["squeue", "-n", job_name, "-h", "-o", "%T"])
    if not output:
        return None
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return None
    return lines[0]


def _sacct_last_state(job_name: str) -> Optional[str]:
    output = _run(["sacct", "-n", "--name", job_name, "-o", "JobID,State", "-X"])
    if not output:
        return None
    last_state = None
    last_job = -1
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        job_id = parts[0].split(".")[0]
        if not job_id.isdigit():
            continue
        job_num = int(job_id)
        if job_num >= last_job:
            last_job = job_num
            last_state = parts[1].split("+")[0]
    return last_state


def _submit(sbatch_script: Path) -> Optional[str]:
    output = _run(["sbatch", sbatch_script.as_posix()])
    for token in output.split():
        if token.isdigit():
            return token
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor and resubmit a Slurm job by name.")
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--sbatch-script", type=Path, required=True)
    parser.add_argument("--interval-mins", type=int, default=10)
    parser.add_argument("--max-hours", type=float, default=10.0)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--eval-command", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(args.log_file.as_posix())
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        LOGGER.addHandler(handler)

    start = time.time()
    interval = max(1, int(args.interval_mins)) * 60
    LOGGER.info("Monitoring job '%s' for up to %.2f hours", args.job_name, args.max_hours)

    while True:
        elapsed_hours = (time.time() - start) / 3600.0
        if elapsed_hours >= float(args.max_hours):
            LOGGER.warning("Stopping after %.2f hours without completion", elapsed_hours)
            return

        state = _squeue_state(args.job_name)
        if state:
            LOGGER.info("Job %s state=%s", args.job_name, state)
            time.sleep(interval)
            continue

        last_state = _sacct_last_state(args.job_name)
        LOGGER.info("Last state for %s: %s", args.job_name, last_state or "unknown")
        if last_state is None:
            job_id = _submit(args.sbatch_script)
            LOGGER.info("Submitted %s as job %s", args.sbatch_script, job_id or "unknown")
            time.sleep(interval)
            continue

        if last_state in RESUBMIT_STATES:
            job_id = _submit(args.sbatch_script)
            LOGGER.info("Resubmitted %s as job %s", args.sbatch_script, job_id or "unknown")
            time.sleep(interval)
            continue

        if last_state == "COMPLETED":
            LOGGER.info("Job %s completed", args.job_name)
            if args.eval_command:
                LOGGER.info("Running eval command: %s", args.eval_command)
                subprocess.run(args.eval_command, shell=True, check=False)
            return

        if last_state in TERMINAL_STATES:
            LOGGER.warning("Job %s ended with state %s", args.job_name, last_state)
            if last_state in RESUBMIT_STATES:
                job_id = _submit(args.sbatch_script)
                LOGGER.info("Resubmitted %s as job %s", args.sbatch_script, job_id or "unknown")
            time.sleep(interval)
            continue

        time.sleep(interval)


if __name__ == "__main__":
    main()
