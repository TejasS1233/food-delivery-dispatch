# Food Delivery Dispatch OpenEnv (v2)

This environment models a realistic food-delivery dispatch system with:
- stochastic order arrivals,
- preparation delays,
- traffic multipliers,
- assignment/rejection/reposition decisions,
- and KPI-based grading.

## Why This Is Real-World

The objective is operationally meaningful: improve on-time delivery while controlling cancellations, rejections, and courier inefficiency.

## Task Set

- `easy`: stable demand, relaxed SLA
- `medium`: stronger peaks and tighter SLA
- `hard`: high volatility, strict SLA, severe peak traffic

## Action Space

`FoodDeliveryAction`
- `assign(order_id, courier_id)`
- `reject(order_id)`
- `reposition(courier_id, target_zone)`
- `wait()`

## Observation Space

`FoodDeliveryObservation`
- top pending orders with urgency/prep/SLA signals
- courier status (zone, busy time, queue size)
- system counters (delivered, rejected, cancelled, on-time)
- per-step reward and cumulative reward

## Reward Design

Dense reward every step:
- positive for on-time deliveries
- smaller positive for late deliveries
- penalty for late minutes
- penalty for cancellations and invalid actions
- small movement/idle penalties to discourage waste

## Grader

`/grader` returns score in `[0,1]` based on:
- on-time rate
- delivery speed
- cancellation/rejection rates
- utilization proxy
- fairness proxy

## Endpoints

- `POST /reset`
- `POST /step`
- `GET /tasks`
- `POST /grader`
- `POST /baseline`
- `POST /evaluate`
- `GET /policies`
- `GET /health`

`/evaluate` also supports `policy_id="auto_best"` which routes:
- `easy`/`medium` -> `ddqn_per_v1` (if registered)
- `hard` -> `hybrid`

## Quick Start

```bash
cd food_delivery_env_v2
uv sync
uv run python -m server.app
```

If you want to run learned-policy training/evaluation, install training extras:

```bash
uv sync --extra train
```

For tests:

```bash
uv sync --extra dev
```

In another terminal:

```bash
uv run python run_baseline.py --url http://localhost:8000 --episodes 3
```

## Submission Utilities

### Inference Script (required)

File: `inference.py` (project root)

Set LLM env vars (Groq example via OpenAI client):

```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
export HF_TOKEN="<your_groq_or_hf_key>"
export BENCHMARK_URL="http://localhost:8000"
export TASK_NAME="medium"
```

Run:

```bash
uv run python inference.py
```

### Pre-validation script

File: `validate-submission.sh`

```bash
chmod +x validate-submission.sh
./validate-submission.sh https://your-space.hf.space .
```

## Notes

- This repo is intentionally independent from `my_env/`.
- Trained policy registry is stored in `models/registry.json`.

## Training Variants

Train DDQN+PER-style meta-policy:

```bash
uv run python train_ddqn_per.py --episodes 300 --policy-id ddqn_per_v1
```

Train masked PPO meta-policy:

```bash
uv run python train_ppo_masked.py --updates 120 --policy-id ppo_masked_v1
```

Alternative training options you can add on top of this repo:
- distributional DQN (C51 / QR-DQN)
- SAC-discrete style policy optimization
- imitation warm-start (behavior cloning on heuristic trajectories) + RL fine-tuning
- offline RL from logged dispatch traces (CQL/IQL style)
- constrained RL with explicit cancellation or SLA violation budgets

Evaluate any policy (heuristic or trained):

```bash
curl -X POST http://localhost:8000/evaluate -H "Content-Type: application/json" -d '{"task_id":"hard","policy_id":"ppo_masked_v1","episodes":5}'
```

Benchmark snapshots are tracked in `RESULTS.md`.
