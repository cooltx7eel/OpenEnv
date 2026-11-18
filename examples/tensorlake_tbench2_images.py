#!/usr/bin/env python3
"""Register one Tensorlake sandbox image for each Terminal-Bench 2 task.

A Terminal-Bench 2 task is scored correctly only in the task's own image: the
image holds the files, the tools and the working directory that the task and
its tests expect. This script takes each task's Dockerfile and adds one layer
with the OpenEnv `tbench2_env` server in `/opt/envserver`. The server has its
own Python, so the layer does not change the task's tools or its `PATH`.

Each image is registered as `tb2-<task>`. A build takes some minutes.

Usage:
    uv run python examples/tensorlake_tbench2_images.py
    uv run python examples/tensorlake_tbench2_images.py fix-git regex-log

Requires:
    The `tl` CLI, logged in (`tl login`) or with TENSORLAKE_API_KEY set.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
TB2_ZIP = (
    "https://github.com/laude-institute/terminal-bench-2/archive/refs/heads/main.zip"
)

# Tasks with small images that an agent can do in some minutes.
DEFAULT_TASKS = [
    "cancel-async-tasks",
    "regex-log",
    "openssl-selfsigned-cert",
    "log-summary-date-ranges",
    "git-leak-recovery",
    "sanitize-git-repo",
    "nginx-request-logging",
    "polyglot-c-py",
]

# Added after the task's own Dockerfile. It sets no WORKDIR and no ENV.
SERVER_LAYER = """
# --- OpenEnv server layer ---
USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
RUN curl -LsSf https://astral.sh/uv/0.9.5/install.sh \\
    | env UV_INSTALL_DIR=/opt/envserver/bin UV_NO_MODIFY_PATH=1 sh
COPY _openenv /opt/envserver/openenv
COPY _env /opt/envserver/env
COPY _task /opt/tb2/{task}
RUN /opt/envserver/bin/uv venv --python 3.11 /opt/envserver/venv \\
    && /opt/envserver/bin/uv pip install --compile-bytecode \\
       --python /opt/envserver/venv/bin/python \\
       /opt/envserver/openenv "camel-ai>=0.2.90"
"""


def image_name(task: str) -> str:
    return f"tb2-{task}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tasks", nargs="*", default=DEFAULT_TASKS, help="task ids")
    args = parser.parse_args()

    failed = []
    with tempfile.TemporaryDirectory() as tmp:
        # Keep the download, so that a second run does not get it again.
        cached = Path(tempfile.gettempdir()) / "terminal-bench-2.zip"
        if not cached.exists():
            print("Downloading the Terminal-Bench 2 tasks (about 40 MB)...")
            part = cached.with_suffix(".part")
            with urllib.request.urlopen(TB2_ZIP, timeout=60) as response:
                with open(part, "wb") as out:
                    size = 0
                    while chunk := response.read(1 << 20):
                        out.write(chunk)
                        size += len(chunk)
                        print(f"\r  {size >> 20} MB", end="", flush=True)
            print()
            part.rename(cached)
        archive = zipfile.ZipFile(cached)
        archive.extractall(tmp)
        tasks_root = Path(tmp) / archive.namelist()[0].split("/")[0]

        for task in args.tasks:
            task_dir = tasks_root / task
            if not (task_dir / "environment" / "Dockerfile").is_file():
                print(f"[{task}] not a Terminal-Bench 2 task, skipped")
                failed.append(task)
                continue

            # The build context is the task's own context plus three folders.
            build = Path(tmp) / "build" / task
            shutil.copytree(task_dir / "environment", build)
            shutil.copytree(task_dir, build / "_task")
            shutil.copytree(
                _REPO_ROOT / "envs" / "tbench2_env",
                build / "_env",
                ignore=shutil.ignore_patterns("__pycache__", ".venv"),
            )
            openenv = build / "_openenv"
            openenv.mkdir()
            for name in ("pyproject.toml", "README.md", "LICENSE"):
                shutil.copy(_REPO_ROOT / name, openenv / name)
            shutil.copytree(
                _REPO_ROOT / "src",
                openenv / "src",
                ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"),
            )
            with open(build / "Dockerfile", "a", encoding="utf-8") as dockerfile:
                dockerfile.write(SERVER_LAYER.format(task=task))

            print(f"[{task}] building image {image_name(task)}...")
            result = subprocess.run(
                ["tl", "sbx", "image", "create", "Dockerfile", "-n", image_name(task)],
                cwd=build,
            )
            if result.returncode != 0:
                print(f"[{task}] build failed")
                failed.append(task)

    done = [t for t in args.tasks if t not in failed]
    print(f"Registered: {' '.join(done) or 'none'}")
    if failed:
        print(f"Failed: {' '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
