---
title: Food Delivery Environment Server
emoji: "🚚"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /docs
tags:
  - openenv
  - logistics
  - reinforcement-learning
---

# Food Delivery Dispatch Environment

Real-world dispatch simulation for food delivery operations, built for OpenEnv and the Meta PyTorch OpenEnv Hackathon.

## Environment Description and Motivation

Dispatch systems for food platforms must make high-frequency decisions under uncertainty: which courier gets which order, when to reject infeasible orders, and where to reposition idle couriers before demand spikes. This environment models those operational trade-offs with stochastic arrivals, prep delays, and traffic multipliers.

The objective is to maximize service quality (on-time delivery, low cancellations) while maintaining efficiency (good utilization, minimal wasted movement).

## Quick Start

```bash
uv sync
uv run python -m server.app
```

In another terminal:

```bash
uv run python scripts/run_baseline.py --url http://localhost:8000 --episodes 3
```

Run tests:

```bash
uv sync --dev
uv run pytest tests/ -v
```

## Task Descriptions and Difficulty

| Task | Description | Couriers | Horizon | SLA | Difficulty |
|---|---|---:|---:|---:|---|
| `easy` | Stable demand, relaxed constraints | 12 | 180 min | 38 min | Intro / low volatility |
| `medium` | Burstier demand, tighter SLA | 14 | 240 min | 34 min | Moderate operational pressure |
| `hard` | Adversarial peaks, strict SLA | 14 | 300 min | 30 min | High volatility and constraint pressure |

All tasks use fixed seeds for deterministic evaluation.

## Action Space Definition

Action type: `FoodDeliveryAction`

| Field | Type | Description |
|---|---|---|
| `action_type` | enum (`assign`, `reject`, `reposition`, `wait`) | Dispatch operation |
| `order_id` | string or null | Order target for assign/reject |
| `courier_id` | string or null | Courier target for assign/reposition |
| `target_zone` | integer or null | Zone id for reposition |

## Observation Space Definition

Observation type: `FoodDeliveryObservation`

Includes complete dispatch state and KPI counters:
- time (`minute`, `horizon`)
- `pending_orders` (urgency, ETA, SLA remaining)
- `couriers` (zone, queue, busy/reposition state)
- totals: created, delivered, on-time, late, rejected, cancelled
- `average_delivery_minutes`
- step `reward`, `cumulative_reward`, and `done`

See `models.py` for the exact typed schema.

## Reward Design

Dense reward is applied every step:
- On-time delivery: `+2.0`
- Late delivery: `+0.8 - 0.03 * late_minutes`
- Assignment shaping bonus: `+0.02 * min(15, slack)`
- Reject: `-0.25`
- Auto-cancel overdue: `-1.2`
- Invalid action: `-0.5`
- Idle courier penalty: `-0.003` per idle courier
- Reposition cost: `-0.02 * travel_time`
- Wait with pending orders: `-0.01 * pending_count`

## Grader

`/grader` returns deterministic score in `[0, 1]` from weighted service and efficiency metrics.

## OpenEnv and Hackathon Endpoints

Standard OpenEnv routes:
- `POST /reset`
- `POST /step`
- `GET /state`

Hackathon-relevant routes:
- `GET /tasks`
- `POST /grader`
- `GET /baseline`
- `POST /baseline`
- `POST /evaluate`
- `GET /health`

## Baseline Scores (Policy Benchmarks)

| Task | Policy | Score | On-Time | Cancel | Avg Delivery |
|---|---|---:|---:|---:|---:|
| easy | nearest | 0.858 | 98.9% | 0.0% | 18.2 min |
| easy | hybrid | 0.863 | 100.0% | 0.0% | 18.0 min |
| easy | ddqn_per_v1 | 0.905 | 98.8% | 0.0% | 18.5 min |
| medium | hybrid | 0.807 | 92.9% | 1.1% | 23.1 min |
| medium | ddqn_per_v1 | 0.864 | 93.3% | 0.0% | 22.2 min |
| hard | hybrid | 0.698 | 71.1% | 7.2% | 25.8 min |
| hard | ddqn_per_v1 | 0.639 | 57.1% | 2.4% | 30.1 min |

## LLM Inference Results (Our Runs)

Using `inference.py` with model `llama-3.3-70b-versatile`:

| Task | Horizon | Steps | Done | Score |
|---|---:|---:|---|---:|
| easy | 180 | 180 | true | 0.924 |
| medium | 240 | 240 | true | 0.781 |
| hard | 300 | 300 | true | 0.712 |

These runs are deterministic for the same task/model/environment settings due to fixed scenario seeds.

## Inference Script (Submission Path)

Root `inference.py` is the script used by evaluators.
- Uses OpenAI-compatible client
- Emits strict logs: `[START]`, repeated `[STEP]`, `[END]`
- Supports single-task execution via `TASK_NAME`
- Uses env horizon by default (`MAX_STEPS=0` means auto-horizon)

Required environment variables:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Optional variables:
- `BENCHMARK_URL` (default `http://localhost:8000`)
- `TASK_NAME` (`easy` | `medium` | `hard`, default `medium`)
- `MAX_STEPS` (`0` for auto-horizon, or positive integer override)

### Reproduce Inference Runs

Reference configuration used by the project team for reported runs: OpenAI-compatible endpoint with model `llama-3.3-70b-versatile`.
Evaluators may use a different endpoint/model; only the required variable names must match.

macOS/Linux (bash):

```bash
export API_BASE_URL="<openai_compatible_base_url>"
export MODEL_NAME="<evaluator_model_name>"
export HF_TOKEN="<your_key>"
export BENCHMARK_URL="<env_base_url>"
export MAX_STEPS="0"

export TASK_NAME="easy"
uv run --no-sync python inference.py

export TASK_NAME="medium"
uv run --no-sync python inference.py

export TASK_NAME="hard"
uv run --no-sync python inference.py
```

Windows (PowerShell):

```powershell
$env:API_BASE_URL="<openai_compatible_base_url>"
$env:MODEL_NAME="<evaluator_model_name>"
$env:HF_TOKEN="<your_key>"
$env:BENCHMARK_URL="<env_base_url>"
$env:MAX_STEPS="0"

$env:TASK_NAME="easy";   uv run --no-sync python inference.py
$env:TASK_NAME="medium"; uv run --no-sync python inference.py
$env:TASK_NAME="hard";   uv run --no-sync python inference.py
```

## How A Judge / LLM Evaluator Will Run This

Typical evaluation flow in the hackathon:
1. Start the environment (local Docker or deployed HF Space).
2. Verify service endpoints (`/health`, `/reset`, `/step`, `/tasks`, `/grader`).
3. Set `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`, `BENCHMARK_URL`, and `TASK_NAME`.
4. Run `uv run --no-sync python inference.py` per task.
5. Parse `[END]` for final `success`, `steps`, and `score`.
6. Confirm full-horizon completion (`done=true` at task horizon).

Example judge run (against deployed HF Space):

macOS/Linux (bash):

```bash
export BENCHMARK_URL="https://tejass1233-food-delivery.hf.space"
export API_BASE_URL="<openai_compatible_base_url>"
export MODEL_NAME="<evaluator_model_name>"
export HF_TOKEN="<judge_key>"
export MAX_STEPS="0"

export TASK_NAME="easy" && uv run --no-sync python inference.py
export TASK_NAME="medium" && uv run --no-sync python inference.py
export TASK_NAME="hard" && uv run --no-sync python inference.py
```

Windows (PowerShell):

```powershell
$env:BENCHMARK_URL="https://tejass1233-food-delivery.hf.space"
$env:API_BASE_URL="<openai_compatible_base_url>"
$env:MODEL_NAME="<evaluator_model_name>"
$env:HF_TOKEN="<judge_key>"
$env:MAX_STEPS="0"

$env:TASK_NAME="easy";   uv run --no-sync python inference.py
$env:TASK_NAME="medium"; uv run --no-sync python inference.py
$env:TASK_NAME="hard";   uv run --no-sync python inference.py
```

Validator command used before final submission:

```bash
bash ./validate-submission.sh https://tejass1233-food-delivery.hf.space .
```

Windows PowerShell (runs the same script through Git Bash):

```powershell
bash ./validate-submission.sh https://tejass1233-food-delivery.hf.space .
```

## Docker

```bash
docker build -t food-delivery-dispatch:latest .
docker run --rm -p 8000:8000 food-delivery-dispatch:latest
curl http://localhost:8000/health
curl http://localhost:8000/baseline
```

`server/Dockerfile` is a legacy alternate. Root `Dockerfile` is canonical.

## Hugging Face Spaces

```bash
openenv push
```

After deploy, verify:
- `POST /reset` returns `200`
- `/tasks`, `/grader`, `/baseline` return valid responses
- Space variables include `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`

## Project Structure

```text
food_delivery_env_v2/
├── README.md
├── openenv.yaml
├── pyproject.toml
├── Dockerfile
├── inference.py
├── client.py
├── models.py
├── decision.py
├── scripts/
│   └── run_baseline.py
├── server/
│   ├── app.py
│   ├── food_delivery_environment.py
│   ├── grader.py
│   ├── web_ui.py
│   └── Dockerfile
├── training/
│   ├── common.py
│   ├── inference.py
│   ├── train_ddqn_per.py
│   ├── train_ppo_masked.py
│   └── models/
│       ├── registry.json
│       ├── ddqn_per_v1.pt
│       └── ppo_masked_v1.pt
├── tests/
│   ├── conftest.py
│   ├── test_api_endpoints.py
│   └── test_grader_and_env.py
└── docs/
    ├── benchmarks/
    ├── submission/
    └── hackathon-docs/
```
