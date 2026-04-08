# OpenEnv Workflow Tutorial

This is the shortest practical path to understand OpenEnv and use it correctly.

## What is happening

OpenEnv turns an RL environment into a service:

1. You call a typed Python client (`reset`, `step`, `state`).
2. The client talks to an environment server (usually over WebSocket `/ws`).
3. The server runs isolated logic in a container/process.
4. You can run this locally, in Docker, or on Hugging Face Spaces.

Core idea: same API for development and production, with better isolation and reproducibility than in-process environments.

## What to do (standard workflow)

### 1) Pick your starting point

- Use an existing environment if available (fastest).
- Create a new one with `openenv init <env_name>` if needed.

### 2) Install and connect

Install client package:

```bash
uv pip install git+https://huggingface.co/spaces/openenv/echo-env
```

Use hosted Space (quick start):

```python
import asyncio
from echo_env import EchoEnv, EchoAction

async def main():
    async with EchoEnv(base_url="https://openenv-echo-env.hf.space") as env:
        await env.reset()
        await env.step(EchoAction(message="Hello"))

asyncio.run(main())
```

### 3) Move to local runtime for reliability/speed

Option A: run server directly for iteration:

```bash
uv sync
uv run server
```

Option B: run Docker image for reproducible local execution:

```bash
docker pull registry.hf.space/openenv-echo-env:latest
docker run -d -p 8000:8000 registry.hf.space/openenv-echo-env:latest
```

Then connect your client to `http://localhost:8000`.

### 4) Deploy

```bash
openenv push --repo-id <username>/<env-name>
```

After deploy, verify:

- `https://<username>-<env-name>.hf.space/health`
- `https://<username>-<env-name>.hf.space/docs`
- `https://<username>-<env-name>.hf.space/web`

### 5) Scale when load grows

Start with one container and tune:

- `WORKERS` (process parallelism)
- `MAX_CONCURRENT_ENVS` (session limit)

Example:

```bash
docker run -d -p 8000:8000 -e WORKERS=8 -e MAX_CONCURRENT_ENVS=400 <image>
```

If a single container saturates, run multiple containers behind a WebSocket-capable load balancer.

### 6) Train with RL

- Connect trainer to the OpenEnv client.
- Rollout loop: generate action -> `env.step()` -> collect reward/feedback -> optimize policy.
- For higher concurrency training, run local Docker or your own infra instead of relying only on hosted free-tier Spaces.

## Quick decisions

- Fast demo: hosted HF Space.
- Active development: local Uvicorn.
- Reproducible runs: local Docker.
- High throughput: multi-container + load balancer.

## Minimal checklist

- Environment responds on `/health`.
- Client `reset` and `step` both succeed.
- Typed action/observation models are stable.
- Runtime variables (`WORKERS`, `MAX_CONCURRENT_ENVS`) are set for expected load.
- Training loop can complete episodes without connection drops.
