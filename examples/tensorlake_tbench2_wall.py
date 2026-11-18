#!/usr/bin/env python3
"""Evaluation wall: an LLM agent does Terminal-Bench 2 tasks on Tensorlake.

Each task runs in its own Tensorlake sandbox, started with
``TensorlakeProvider``. A model on OpenRouter works on the task with shell commands through
the OpenEnv ``tbench2_env`` client. Then the environment scores the result with
the task's own tests. The browser shows one live terminal for each task.

When a task fails, the script forks the failed sandbox: it makes live copies,
gives each copy the test output, and lets the agent try again in all the copies
at the same time. A copy keeps the files, the processes and the memory of the
source sandbox, so no task setup runs again.

Register the task images first:

    uv run python examples/tensorlake_tbench2_images.py

Usage:
    PYTHONPATH=src:envs uv run python examples/tensorlake_tbench2_wall.py

Then open http://127.0.0.1:8080. Press Ctrl+C to stop all the sandboxes.

Requires:
    TENSORLAKE_API_KEY and OPENROUTER_API_KEY environment variables.
"""

import argparse
import asyncio
import html
import json
import logging
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "envs"))

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from openai import AsyncOpenAI, AuthenticationError, NotFoundError
from openenv.core.containers.runtime.tensorlake_provider import TensorlakeProvider
from tbench2_env import Tbench2Action, Tbench2Env
from tensorlake.sandbox import sandbox_url_from_ingress_endpoint
from tensorlake_tbench2_images import DEFAULT_TASKS, image_name

logger = logging.getLogger(__name__)

PORT = 8000
FRAME_DELAY_S = 0.25
TERMINAL_LINES = 60  # lines kept for each terminal on the page
OUTPUT_LINES = 12  # lines shown for one command
MAX_OUTPUT_CHARS = 8000  # command output given to the model

# The image from tensorlake_tbench2_images.py keeps the server outside the task
# tree, so the provider cannot find the command in /app/env/openenv.yaml.
SERVER_CMD = (
    "cd /opt/envserver/env && exec /opt/envserver/venv/bin/python "
    "-m uvicorn server.app:app --host 0.0.0.0 --port 8000"
)
SERVER_ENV = {
    "TB2_TASKS_DIR": "/opt/tb2",
    # Keep the tests in server memory only, where the agent cannot read them.
    "TB2_WITHHOLD_TESTS": "1",
    "TB2_COMMAND_TIMEOUT_S": "300",
}

SYSTEM_PROMPT = """You are an expert engineer who works in a Linux terminal.

You get one task. Work step by step: examine the environment, make changes, \
and check your work with commands.

- Run shell commands with the `run_command` tool. Each command runs \
non-interactively in a new shell in the task's working directory, and returns \
stdout and stderr together.
- No person is available, so do not ask questions.
- When the task is complete, call `submit`. The task's own tests then score \
your work."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a bash command in the task sandbox and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command."}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Declare that the task is complete. The work is then scored.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# Each fork gets a different hint, so that the forks try different fixes.
FORK_HINTS = [
    "Read the test output closely and fix only what failed.",
    "Read the task statement again and check each requirement, one at a time.",
    "Assume that your first approach was wrong, and try a different approach.",
    "Before you submit, write your own checks for each requirement and run them.",
]


def new_terminal(name: str) -> dict:
    return {"name": name, "status": "waiting", "lines": [], "steps": 0, "ready_s": None}


def emit(term: dict, kind: str, text: str) -> None:
    """Add lines to a terminal on the page. `kind` is cmd, out or note."""
    lines = text.rstrip().splitlines() or [""]
    if kind == "out" and len(lines) > OUTPUT_LINES:
        lines = [f"... {len(lines) - OUTPUT_LINES} more lines"] + lines[-OUTPUT_LINES:]
    term["lines"].extend([kind, line[:200]] for line in lines)
    del term["lines"][:-TERMINAL_LINES]


async def run_agent(
    env: Tbench2Env,
    llm: AsyncOpenAI,
    args: argparse.Namespace,
    messages: list,
    term: dict,
    max_steps: int,
) -> None:
    """Let the model run commands until it submits or uses all its steps.

    `messages` grows in place, so a fork can continue from the same history.
    """
    for _ in range(max_steps):
        response = await llm.chat.completions.create(
            model=args.model, messages=messages, tools=TOOLS
        )
        message = response.choices[0].message
        # Keep all the fields of the reply: the provider can need its
        # reasoning fields in the next request.
        messages.append(message.model_dump(exclude_none=True))
        if not message.tool_calls:
            # The model stopped but did not submit. Score what is there.
            return

        submitted = False
        for call in message.tool_calls:
            if call.function.name == "submit":
                submitted = True
                output = "Submitted for scoring."
            else:
                try:
                    command = json.loads(call.function.arguments)["command"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    command = ""
                emit(term, "cmd", command)
                step = await env.step(
                    Tbench2Action(action_type="exec", command=command)
                )
                obs = step.observation
                output = obs.output if obs.success else f"ERROR: {obs.error}"
                emit(term, "out", output)
                term["steps"] += 1
                if len(output) > MAX_OUTPUT_CHARS:
                    cut = len(output) - MAX_OUTPUT_CHARS
                    output = f"{output[:MAX_OUTPUT_CHARS]}\n... [{cut} more characters]"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": output or "(no output)",
                }
            )
        if submitted:
            return
    emit(term, "note", "Step limit reached.")


async def score(env: Tbench2Env, term: dict) -> str:
    """Run the task's tests. Return the test output; set the terminal status."""
    term["status"] = "scoring"
    emit(term, "note", "Scoring with the task's tests...")
    result = await env.step(Tbench2Action(action_type="evaluate"))
    obs = result.observation
    if not obs.success:
        term["status"] = "error"
        emit(term, "note", f"Scoring error: {obs.error[:300]}")
        return ""
    passed = result.reward == 1.0
    term["status"] = "passed" if passed else "failed"
    emit(term, "note", "PASS" if passed else "FAIL")
    return obs.output or ""


def wait_reachable(url: str, deadline: float) -> None:
    """Poll `/health` of a copy until it answers."""
    while time.monotonic() < deadline:
        try:
            if requests.get(url + "/health", timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.25)
    raise TimeoutError("The copy did not answer on /health.")


async def run_branch(
    branch: dict,
    copy,
    client,
    task: str,
    messages: list,
    llm: AsyncOpenAI,
    args: argparse.Namespace,
    fork_started: float,
) -> None:
    """Continue the agent in one copy of the failed sandbox."""
    try:
        deadline = time.monotonic() + 120
        while copy.status == "pending" and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
            copy = await asyncio.to_thread(client.get, copy.sandbox_id)
        if copy.status != "running":
            raise RuntimeError(f"copy status is {copy.status}")
        url = sandbox_url_from_ingress_endpoint(
            copy.ingress_endpoint, copy.sandbox_id, PORT
        )
        await asyncio.to_thread(wait_reachable, url, deadline)

        async with Tbench2Env(base_url=url, message_timeout_s=1200) as env:
            # A copy does not keep the WebSocket session of its source. A new
            # session and a reset attach to the copied files: with a task
            # image, reset changes no task file. The withheld tests are in the
            # copied server memory.
            await env.reset(task_id=task)
            # From the copy call to an environment that accepts commands.
            branch["ready_s"] = round(time.monotonic() - fork_started, 1)
            branch["status"] = "working"
            await run_agent(env, llm, args, messages, branch, args.retry_steps)
            await score(env, branch)
    except Exception as exc:
        # Client errors can quote the sandbox URL, so show only the error type.
        branch["status"] = "error"
        emit(branch, "note", f"Stopped: {type(exc).__name__}")


async def run_tile(tile: dict, llm: AsyncOpenAI, args: argparse.Namespace) -> None:
    """Start one sandbox, do one task in it, score it, and fork it on failure."""
    task = tile["name"]
    # public=True: anyone with the sandbox URL can reach the server while it runs.
    # The idle timeout stops a sandbox that this script could not stop.
    provider = TensorlakeProvider(
        public=True,
        image=image_name(task),
        cmd=SERVER_CMD,
        env_vars=SERVER_ENV,
        cpus=1,
        memory_mb=2048,
        timeout_secs=900,
        # If the server exits, show the end of its output (secrets are masked).
        surface_server_logs=True,
    )
    copy_ids: list[str] = []
    try:
        started = time.monotonic()
        tile["status"] = "booting"
        try:
            base_url = await asyncio.to_thread(provider.start_container)
            tile["sandbox_s"] = round(time.monotonic() - started, 1)
            await asyncio.to_thread(provider.wait_for_ready, base_url, 300)
        except Exception as exc:
            # Provider errors carry no SDK details, so the message is safe to show.
            tile["status"] = "error"
            emit(tile, "note", f"Sandbox did not start: {exc}")
            logger.error("Task %s: sandbox did not start: %s", task, exc)
            return
        tile["server_s"] = round(time.monotonic() - started - tile["sandbox_s"], 1)

        async with Tbench2Env(base_url=base_url, message_timeout_s=1200) as env:
            reset_started = time.monotonic()
            result = await env.reset(task_id=task)
            tile["reset_s"] = round(time.monotonic() - reset_started, 1)
            # From the create call to an environment that accepts commands.
            tile["start_s"] = round(time.monotonic() - started, 1)
            instruction = result.observation.instruction
            emit(tile, "note", "Task: " + " ".join(instruction.split())[:300])
            tile["status"] = "working"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
            ]
            await run_agent(env, llm, args, messages, tile, args.max_steps)
            test_output = await score(env, tile)
        tile["first"] = tile["status"]
        if tile["status"] != "failed" or args.forks == 0:
            return

        # The provider has no copy method, so use its SDK client directly.
        client = provider._client
        age_s = round(time.monotonic() - started)
        fork_started = time.monotonic()
        copied = await asyncio.to_thread(
            client.copy, provider._sandbox_id, times=args.forks, request_timeout=300
        )
        copy_ids.extend(c.sandbox_id for c in copied.sandboxes)
        tile["fork_s"] = round(time.monotonic() - fork_started, 1)
        emit(
            tile,
            "note",
            f"Forked {args.forks} times in {tile['fork_s']} s. Each copy has the "
            f"state that this sandbox made in {age_s} s.",
        )

        branches = []
        for i, copy in enumerate(copied.sandboxes):
            branch = new_terminal(f"fork {i + 1}")
            tile["branches"].append(branch)
            feedback = (
                "The tests failed. This is the end of the test output:\n\n"
                f"{test_output[-3000:]}\n\n"
                f"Fix the problem, then call `submit` again. "
                f"{FORK_HINTS[i % len(FORK_HINTS)]}"
            )
            history = [*messages, {"role": "user", "content": feedback}]
            branches.append(
                run_branch(branch, copy, client, task, history, llm, args, fork_started)
            )
        await asyncio.gather(*branches)
    except Exception as exc:
        # Client errors can quote the sandbox URL, so show only the error type.
        tile["status"] = "error"
        emit(tile, "note", f"Stopped: {type(exc).__name__}")
        logger.error("Task %s stopped: %s", task, type(exc).__name__)
    finally:
        for copy_id in copy_ids:
            try:
                await asyncio.to_thread(provider._client.delete, copy_id)
            except Exception as exc:
                logger.error("Could not delete a copy: %s", type(exc).__name__)
        await asyncio.to_thread(provider.stop_container)


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evaluation wall</title>
<style>
  :root { --bg:#0b0e14; --panel:#141923; --line:#232a38; --text:#e6e9ef; --dim:#8590a6;
          --ok:#3ddc97; --bad:#ff5d73; --warn:#f5b942; --cmd:#8ab4ff; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:13px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; }
  header { display:flex; flex-wrap:wrap; gap:12px 40px; align-items:center;
           padding:14px 20px; border-bottom:1px solid var(--line); }
  h1 { font-size:16px; margin:0 auto 0 0; font-weight:600; }
  h1 small { display:block; margin-top:2px; font-size:11px; font-weight:400; color:var(--dim); }
  .stat b { display:block; font-size:22px; line-height:1.2; color:var(--ok);
            font-variant-numeric:tabular-nums; }
  .stat span { color:var(--dim); font-size:11px; }
  #wall { display:grid; gap:12px; padding:16px 20px;
          grid-template-columns:repeat(auto-fill, minmax(420px, 1fr)); }
  .term { background:var(--panel); border:1px solid var(--line); border-radius:8px;
          padding:8px; min-width:0; transition:border-color .4s;
          display:flex; flex-direction:column; }
  .term.working, .term.scoring { border-color:#3b4a6b; }
  .term.passed { border-color:var(--ok); }
  .term.failed, .term.error { border-color:var(--bad); }
  .bar { display:flex; justify-content:space-between; gap:12px; margin-bottom:6px; }
  .bar .name { font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .bar .status { color:var(--warn); white-space:nowrap; }
  .passed > .bar .status { color:var(--ok); }
  .failed > .bar .status, .error > .bar .status { color:var(--bad); }
  pre { margin:0; flex:1 1 220px; min-height:220px; overflow:hidden; background:var(--bg); border-radius:4px;
        padding:6px 8px; font:11px/1.35 inherit; white-space:pre-wrap; word-break:break-all;
        display:flex; flex-direction:column; justify-content:flex-end; }
  pre .cmd { color:var(--cmd); } pre .cmd::before { content:"$ "; color:var(--dim); }
  pre .out { color:var(--dim); } pre .note { color:var(--warn); }
  .forks { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:8px; }
  .forks pre { flex:none; min-height:0; height:130px; font-size:10px; }
</style>
</head>
<body>
<header>
  <h1>OpenEnv on Tensorlake: Terminal-Bench 2<small>agent model: __MODEL__</small></h1>
  <div class="stat"><b id="envready">-</b><span>avg environment ready</span></div>
  <div class="stat"><b id="boot">-</b><span>avg sandbox start</span></div>
  <div class="stat"><b id="ready">0</b><span>sandboxes ready</span></div>
  <div class="stat"><b id="first">0</b><span>passed, first attempt</span></div>
  <div class="stat"><b id="final">0</b><span>passed, with fork retries</span></div>
  <div class="stat"><b id="fork">-</b><span>avg fork start</span></div>
</header>
<main id="wall"></main>
<script>
const wall = document.getElementById('wall');
const terms = new Map();

function makeTerm(key, parent) {
  const el = document.createElement('div');
  const bar = document.createElement('div');
  bar.className = 'bar';
  const name = document.createElement('span');
  name.className = 'name';
  const status = document.createElement('span');
  status.className = 'status';
  bar.append(name, status);
  const pre = document.createElement('pre');
  const body = document.createElement('div');
  pre.append(body);
  el.append(bar, pre);
  parent.append(el);
  const term = {el, name, status, body, count: -1, forks: null};
  terms.set(key, term);
  return term;
}

function update(key, parent, t, label) {
  const term = terms.get(key) || makeTerm(key, parent);
  term.el.className = 'term ' + t.status;
  term.name.textContent = t.name;
  term.status.textContent = label;
  const count = t.steps * 1000 + t.lines.length;
  if (count !== term.count) {
    term.count = count;
    term.body.replaceChildren(...t.lines.map(([kind, text]) => {
      const line = document.createElement('div');
      line.className = kind;
      line.textContent = text;
      return line;
    }));
  }
  return term;
}

function average(values) {
  if (!values.length) return '-';
  return (values.reduce((a, b) => a + b, 0) / values.length).toFixed(1) + ' s';
}

new EventSource('/events').onmessage = event => {
  const data = JSON.parse(event.data);
  let ready = 0, first = 0, final = 0;
  const boots = [], envs = [], forks = [];
  for (const t of data) {
    if (t.sandbox_s !== null) boots.push(t.sandbox_s);
    if (t.start_s !== null) { ready += 1; envs.push(t.start_s); }
    const term = update('t' + t.id, wall, t, t.status + ' · ' + t.steps + ' commands');
    if (t.first === 'passed') first += 1;
    let passed = t.status === 'passed';
    if (t.branches.length && !term.forks) {
      term.forks = document.createElement('div');
      term.forks.className = 'forks';
      term.el.append(term.forks);
    }
    t.branches.forEach((b, i) => {
      update('t' + t.id + 'b' + i, term.forks, b, b.status + ' · ' + b.steps);
      if (b.ready_s !== null) forks.push(b.ready_s);
      if (b.status === 'passed') passed = true;
    });
    if (passed) final += 1;
  }
  document.getElementById('ready').textContent = ready + ' / ' + data.length;
  document.getElementById('first').textContent = first + ' / ' + data.length;
  document.getElementById('final').textContent = final + ' / ' + data.length;
  document.getElementById('envready').textContent = average(envs);
  document.getElementById('boot').textContent = average(boots);
  document.getElementById('fork').textContent = average(forks);
};
</script>
</body>
</html>
"""


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tasks", nargs="*", default=DEFAULT_TASKS, help="task ids")
    parser.add_argument(
        "--model", default="anthropic/claude-opus-5.5", help="OpenRouter model id"
    )
    parser.add_argument(
        "--max-steps", type=int, default=30, help="model turns for the first attempt"
    )
    parser.add_argument(
        "--forks", type=int, default=4, help="copies of a failed sandbox; 0 for none"
    )
    parser.add_argument(
        "--retry-steps", type=int, default=15, help="model turns for each fork"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="local port for the page"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # The sandbox URL is a bearer capability: keep httpx from logging it (RFC 002 S2).
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if not os.environ.get("TENSORLAKE_API_KEY"):
        raise SystemExit("Set TENSORLAKE_API_KEY to create the sandboxes.")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("Set OPENROUTER_API_KEY to query the model.")
    llm = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        max_retries=6,
    )
    # Check the key and the model id with one small request, before any
    # sandbox starts.
    try:
        await llm.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Reply with: ok"}],
        )
    except AuthenticationError:
        raise SystemExit(
            "OpenRouter refused the key. Check OPENROUTER_API_KEY."
        ) from None
    except NotFoundError:
        raise SystemExit(f"OpenRouter has no model {args.model!r}.") from None

    # Each sandbox start blocks one thread until the server is ready.
    threads = len(args.tasks) * (1 + args.forks) + 4
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(threads))

    tiles = [
        {
            **new_terminal(task),
            "id": i,
            "start_s": None,
            "sandbox_s": None,
            "server_s": None,
            "reset_s": None,
            "fork_s": None,
            "first": "",
            "branches": [],
        }
        for i, task in enumerate(args.tasks)
    ]
    stop = asyncio.Event()

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    async def page() -> str:
        return PAGE.replace("__MODEL__", html.escape(args.model))

    @app.get("/events")
    async def events() -> StreamingResponse:
        async def stream():
            while not stop.is_set():
                yield f"data: {json.dumps(tiles)}\n\n"
                await asyncio.sleep(FRAME_DELAY_S)

        return StreamingResponse(stream(), media_type="text/event-stream")

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=args.port,
            log_level="warning",
            timeout_graceful_shutdown=1,
        )
    )

    logger.info("Starting %d tasks with model %s...", len(tiles), args.model)
    logger.info("Open http://127.0.0.1:%d  (Ctrl+C stops all sandboxes)", args.port)
    workers = [asyncio.create_task(run_tile(t, llm, args)) for t in tiles]

    async def report() -> None:
        await asyncio.gather(*workers, return_exceptions=True)
        first = sum(t["first"] == "passed" for t in tiles)
        final = sum(
            t["status"] == "passed"
            or any(b["status"] == "passed" for b in t["branches"])
            for t in tiles
        )
        logger.info("First attempt: %d / %d passed.", first, len(tiles))
        forked = sum(bool(t["branches"]) for t in tiles)
        if forked:
            logger.info("Failed tasks forked: %d.", forked)
            logger.info("With fork retries: %d / %d passed.", final, len(tiles))
        else:
            logger.info("No task failed, so no sandbox was forked.")
        ready = [t["start_s"] for t in tiles if t["start_s"] is not None]
        if ready:
            logger.info("Average environment ready: %.1f s.", statistics.mean(ready))
        boots = [t["sandbox_s"] for t in tiles if t["sandbox_s"] is not None]
        if boots:
            logger.info("Average sandbox start: %.1f s.", statistics.mean(boots))
        copies = [
            b["ready_s"]
            for t in tiles
            for b in t["branches"]
            if b["ready_s"] is not None
        ]
        if copies:
            logger.info("Average fork start: %.1f s.", statistics.mean(copies))
        logger.info("All sandboxes stopped. Press Ctrl+C to close the page.")

    reporter = asyncio.create_task(report())
    try:
        await server.serve()
    finally:
        stop.set()
        # A cancelled worker stops its sandbox and deletes its copies.
        for worker in workers:
            worker.cancel()
        logger.info("Stopping sandboxes...")
        await asyncio.gather(*workers, return_exceptions=True)
        await asyncio.gather(reporter, return_exceptions=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
