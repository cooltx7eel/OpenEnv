#!/usr/bin/env python3
"""Sandbox wall: many Snake games, each in its own Tensorlake sandbox.

Starts N sandboxes at the same time with ``TensorlakeProvider``, plays Snake in
each one with a simple greedy policy, and shows all the boards in a live grid
in the browser. Each tile shows how long its sandbox took to become ready.

Register the sandbox image first. The Dockerfile copies its build context, so
build from a copy of the env with the Dockerfile at the top level:

    cp -r envs/snake_env /tmp/snake-env && cp /tmp/snake-env/server/Dockerfile /tmp/snake-env/
    cd /tmp/snake-env && tl sbx image create Dockerfile -n snake-env

Usage:
    PYTHONPATH=src:envs uv run python examples/tensorlake_snake_wall.py -n 16

Then open http://127.0.0.1:8080. Press Ctrl+C to stop all the sandboxes.

To compare with local Docker, build the same image locally and run the script
a second time with `--provider docker`. Put the two pages side by side:

    docker build -t snake-env -f envs/snake_env/server/Dockerfile envs/snake_env
    PYTHONPATH=src:envs uv run python examples/tensorlake_snake_wall.py -n 16 --provider docker --port 8081

Requires:
    TENSORLAKE_API_KEY environment variable (not for `--provider docker`).
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "envs"))

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from openenv.core.containers.runtime.providers import LocalDockerProvider
from openenv.core.containers.runtime.tensorlake_provider import TensorlakeProvider
from snake_env import SnakeAction, SnakeEnv

logger = logging.getLogger(__name__)

FRAME_DELAY_S = 0.1

# Grid cell codes (value % 10): 0 empty, 1 wall, 2 fruit, 3 head, other = body.
EMPTY, FRUIT, HEAD = 0, 2, 3


def find(grid, code):
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value % 10 == code:
                return (r, c)
    return None


class GreedyPolicy:
    """Move toward the fruit and do not hit walls or the snake body.

    Actions are relative to the snake: 0 straight, 1 turn left, 2 turn right.
    The policy finds the heading from two successive head positions. It also
    checks the result of its first turn, to learn which way "left" is on the grid.
    """

    def __init__(self):
        self.reset()
        self.left_is_ccw = True

    def reset(self):
        self.head = None
        self.heading = None
        self.expected = None

    def _turn(self, heading, action):
        dr, dc = heading
        if action == 0:
            return heading
        ccw = (action == 1) == self.left_is_ccw
        return (-dc, dr) if ccw else (dc, -dr)

    def act(self, grid) -> int:
        head = find(grid, HEAD)
        if head is None:
            return 0
        if self.head is not None and head != self.head:
            self.heading = (head[0] - self.head[0], head[1] - self.head[1])
            if self.expected is not None and self.heading == (
                -self.expected[0],
                -self.expected[1],
            ):
                self.left_is_ccw = not self.left_is_ccw
        self.head = head
        self.expected = None
        if self.heading is None:
            return 0

        fruit = find(grid, FRUIT) or head
        best, best_dist = 0, None
        for action in (0, 1, 2):
            dr, dc = self._turn(self.heading, action)
            r, c = head[0] + dr, head[1] + dc
            if not (0 <= r < len(grid) and 0 <= c < len(grid[0])):
                continue
            if grid[r][c] % 10 not in (EMPTY, FRUIT):
                continue
            dist = abs(fruit[0] - r) + abs(fruit[1] - c)
            if best_dist is None or dist < best_dist:
                best, best_dist = action, dist
        if best != 0:
            self.expected = self._turn(self.heading, best)
        return best


async def run_tile(
    tile: dict,
    args: argparse.Namespace,
    stop: asyncio.Event,
    start_gate: asyncio.Semaphore,
) -> None:
    """Start one sandbox, then play Snake in it until `stop` is set."""
    # The provider is the only difference between local Docker and Tensorlake.
    if args.provider == "docker":
        provider = LocalDockerProvider()
    else:
        # public=True: anyone with the sandbox URL can reach the server while it runs.
        # The idle timeout stops a sandbox that this script could not stop.
        provider = TensorlakeProvider(public=True, timeout_secs=600)
    try:
        try:
            async with start_gate:
                started = time.monotonic()
                base_url = await asyncio.to_thread(provider.start_container, args.image)
            tile["sandbox_s"] = round(time.monotonic() - started, 1)
            tile["status"] = "booting"
            await asyncio.to_thread(provider.wait_for_ready, base_url, 300)
        except Exception as exc:
            # Provider errors carry no SDK details, so the message is safe to show.
            tile.update(status="failed", error=str(exc)[:200])
            logger.error("Sandbox %d failed to start: %s", tile["id"], exc)
            return
        policy = GreedyPolicy()
        async with SnakeEnv(base_url=base_url) as env:
            result = await env.reset()
            # From the create call to an environment that accepts actions.
            tile["start_s"] = round(time.monotonic() - started, 1)
            tile["status"] = "running"
            while not stop.is_set():
                obs = result.observation
                tile["grid"] = obs.grid
                # The server reports `episode_fruits` only when the episode ends,
                # so count fruits from the reward: 1.0 for each fruit.
                if (result.reward or 0) >= 0.5:
                    tile["fruits"] += 1
                if result.done:
                    tile["episodes"] += 1
                    tile["total_fruits"] += tile["fruits"]
                    tile["fruits"] = 0
                    policy.reset()
                    await asyncio.sleep(0.5)
                    result = await env.reset()
                    continue
                result = await env.step(SnakeAction(action=policy.act(obs.grid)))
                tile["steps"] += 1
                await asyncio.sleep(args.step_delay)
    except Exception as exc:
        # Client errors can quote the sandbox URL, so show only the error type.
        tile.update(status="failed", error=type(exc).__name__)
        logger.error("Sandbox %d stopped: %s", tile["id"], type(exc).__name__)
    finally:
        await asyncio.to_thread(provider.stop_container)
        if tile["status"] != "failed":
            tile["status"] = "stopped"


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sandbox wall</title>
<style>
  :root { --bg:#0b0e14; --panel:#141923; --line:#232a38; --text:#e6e9ef; --dim:#8590a6;
          --wall:#2a3244; --snake:#3ddc97; --head:#b6ffdd; --fruit:#ff5d73; --warn:#f5b942; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; }
  header { display:flex; flex-wrap:wrap; gap:8px 32px; align-items:baseline;
           padding:16px 20px; border-bottom:1px solid var(--line); }
  h1 { font-size:16px; margin:0; font-weight:600; }
  .stat b { font-size:20px; color:var(--snake); font-variant-numeric:tabular-nums; }
  .stat span { color:var(--dim); margin-left:6px; }
  #wall { display:grid; gap:10px; padding:16px 20px;
          grid-template-columns:repeat(auto-fill, minmax(150px, 1fr)); }
  .tile { background:var(--panel); border:1px solid var(--line); border-radius:8px;
          padding:8px; opacity:.45; transition:opacity .4s, border-color .4s; }
  .tile.running { opacity:1; border-color:#2f5d4b; }
  .tile.failed { opacity:1; border-color:var(--fruit); }
  canvas { width:100%; aspect-ratio:1; display:block; border-radius:4px; background:var(--bg); }
  .row { display:flex; justify-content:space-between; margin-top:6px; font-size:12px;
         color:var(--dim); font-variant-numeric:tabular-nums; }
  .row .ok { color:var(--snake); } .row .wait { color:var(--warn); } .row .bad { color:var(--fruit); }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="stat"><b id="ready">0</b><span id="total"></span></div>
  <div class="stat"><b id="rate">0</b><span>steps / s</span></div>
  <div class="stat"><b id="fruits">0</b><span>fruits</span></div>
  <div class="stat"><b id="episodes">0</b><span>episodes</span></div>
  <div class="stat"><b id="envready">-</b><span>s avg environment ready</span></div>
  <div class="stat"><b id="boot">-</b><span>s avg sandbox start</span></div>
</header>
<main id="wall"></main>
<script>
const css = getComputedStyle(document.documentElement);
const color = n => css.getPropertyValue(n).trim();
const COLORS = {1: color('--wall'), 2: color('--fruit'), 3: color('--head')};
const BODY = color('--snake');
const wall = document.getElementById('wall');
const tiles = new Map();
let lastSteps = null, lastTime = null;

function makeTile(id) {
  const el = document.createElement('div');
  el.className = 'tile';
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 200;
  const row = document.createElement('div');
  row.className = 'row';
  const left = document.createElement('span');
  const right = document.createElement('span');
  row.append(left, right);
  el.append(canvas, row);
  wall.append(el);
  const tile = {el, ctx: canvas.getContext('2d'), left, right};
  tiles.set(id, tile);
  return tile;
}

function draw(ctx, grid) {
  ctx.clearRect(0, 0, 200, 200);
  if (!grid || !grid.length) return;
  const cell = 200 / Math.max(grid.length, grid[0].length);
  grid.forEach((line, r) => line.forEach((value, c) => {
    if (value === 0) return;
    ctx.fillStyle = COLORS[value % 10] || BODY;
    ctx.fillRect(c * cell + 0.5, r * cell + 0.5, cell - 1, cell - 1);
  }));
}

function average(values) {
  if (!values.length) return '-';
  return (values.reduce((a, b) => a + b, 0) / values.length).toFixed(1);
}

new EventSource('/events').onmessage = event => {
  const data = JSON.parse(event.data);
  let steps = 0, fruits = 0, episodes = 0, ready = 0;
  const boots = [], envs = [];
  for (const t of data) {
    const tile = tiles.get(t.id) || makeTile(t.id);
    tile.el.className = 'tile ' + t.status;
    draw(tile.ctx, t.grid);
    const running = t.status === 'running';
    tile.left.className = running ? 'ok' : t.status === 'failed' ? 'bad' : 'wait';
    tile.left.textContent = running ? 'ready in ' + t.start_s + ' s'
      : t.status === 'failed' ? 'failed: ' + t.error : t.status + '...';
    tile.right.textContent = running ? '🍎 ' + t.fruits : '';
    steps += t.steps; episodes += t.episodes; fruits += t.total_fruits + t.fruits;
    if (t.sandbox_s !== null) boots.push(t.sandbox_s);
    if (running) { ready += 1; envs.push(t.start_s); }
  }
  const now = performance.now();
  if (lastTime !== null && now - lastTime > 900) {
    document.getElementById('rate').textContent =
      Math.round((steps - lastSteps) / ((now - lastTime) / 1000));
    lastSteps = steps; lastTime = now;
  } else if (lastTime === null) { lastSteps = steps; lastTime = now; }
  document.getElementById('ready').textContent = ready;
  document.getElementById('total').textContent = '/ ' + data.length + ' sandboxes ready';
  document.getElementById('fruits').textContent = fruits;
  document.getElementById('episodes').textContent = episodes;
  document.getElementById('envready').textContent = average(envs);
  document.getElementById('boot').textContent = average(boots);
};
</script>
</body>
</html>
"""


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-n", "--num", type=int, default=16, help="number of sandboxes")
    parser.add_argument("--image", default="snake-env", help="registered image name")
    parser.add_argument(
        "--port", type=int, default=8080, help="local port for the page"
    )
    parser.add_argument(
        "--provider",
        choices=["tensorlake", "docker"],
        default="tensorlake",
        help="where the environments run",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.08,
        help="pause between steps in seconds; use 0 to measure throughput",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # The sandbox URL is a bearer capability: keep httpx from logging it (RFC 002 S2).
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.provider == "tensorlake" and not os.environ.get("TENSORLAKE_API_KEY"):
        raise SystemExit("Set TENSORLAKE_API_KEY to create the sandboxes.")

    # Each sandbox start blocks one thread until the server is ready.
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(args.num + 4))

    tiles = [
        {
            "id": i,
            "status": "creating",
            "start_s": None,
            "sandbox_s": None,
            "error": "",
            "grid": [],
            "fruits": 0,
            "total_fruits": 0,
            "episodes": 0,
            "steps": 0,
        }
        for i in range(args.num)
    ]
    stop = asyncio.Event()
    # LocalDockerProvider picks a free port and a time-based name, so parallel
    # starts can collide. Start local containers one at a time.
    start_gate = asyncio.Semaphore(1 if args.provider == "docker" else args.num)
    where = "local Docker" if args.provider == "docker" else "Tensorlake"
    title = f"OpenEnv on {where}: sandbox wall"

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    async def page() -> str:
        return PAGE.replace("__TITLE__", title)

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

    logger.info("Starting %d sandboxes from image %s...", args.num, args.image)
    logger.info("Open http://127.0.0.1:%d  (Ctrl+C stops all sandboxes)", args.port)
    workers = [asyncio.create_task(run_tile(t, args, stop, start_gate)) for t in tiles]
    try:
        await server.serve()
    finally:
        # Do not cancel the workers: each one must stop its own sandbox.
        stop.set()
        logger.info("Stopping sandboxes...")
        await asyncio.gather(*workers, return_exceptions=True)
        logger.info("All sandboxes stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
