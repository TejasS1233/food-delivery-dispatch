---
title: Food Delivery Environment Server
emoji: "\ud83d\udef5"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /docs
tags:
  - openenv
---

# Food Delivery Dispatch Environment

A realistic food-delivery dispatch simulation for training and evaluating AI agents. Models stochastic order arrivals, preparation delays, traffic multipliers, courier queuing, and KPI-based grading — the same operational challenges faced by Swiggy, Zomato, and Uber Eats.

## Why This Is Real-World

Every day, millions of dispatch decisions are made by humans and algorithms. This environment captures the core trade-offs:
- **On-time delivery** vs **courier utilization**
- **Order rejection** (capacity management) vs **customer satisfaction**
- **Proactive repositioning** (anticipating demand) vs **idle movement cost**

The reward signal is dense and multi-objective, providing meaningful feedback throughout each episode — not just a binary end-of-episode score.

---

## Task Set

| Task | Description | Couriers | Horizon | SLA | Traffic Peaks |
|---|---|---|---|---|---|
| **easy** | Stable demand, relaxed SLA | 12 | 180 min | 38 min | Low (1.1x) |
| **medium** | Peak variability, tighter SLA | 14 | 240 min | 34 min | Moderate (1.25x) |
| **hard** | Adversarial peak, strict SLA | 14 | 300 min | 30 min | Severe (1.45x) |

Each task uses a fixed random seed for deterministic, reproducible evaluation.

---

## Action Space

`FoodDeliveryAction` — one decision per step:

| Field | Type | Description |
|---|---|---|
| `action_type` | `"assign" \| "reject" \| "reposition" \| "wait"` | Dispatch action to apply |
| `order_id` | `str \| None` | Order to assign or reject |
| `courier_id` | `str \| None` | Courier to assign or reposition |
| `target_zone` | `int \| None` | Target zone for repositioning |

---

## Observation Space

`FoodDeliveryObservation` — full system state each step:

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | Task difficulty identifier |
| `minute` | `int` | Current minute of episode |
| `horizon` | `int` | Episode horizon in minutes |
| `pending_orders` | `list[PendingOrderView]` | Top pending orders with urgency, prep time, SLA remaining |
| `couriers` | `list[CourierView]` | Courier status: zone, busy time, queue length |
| `total_orders_created` | `int` | Cumulative orders spawned |
| `total_delivered` | `int` | Successfully delivered orders |
| `total_on_time` | `int` | On-time deliveries |
| `total_late` | `int` | Late deliveries |
| `total_rejected` | `int` | Rejected orders |
| `total_cancelled` | `int` | Auto-cancelled (overdue) orders |
| `average_delivery_minutes` | `float` | Mean delivery duration |
| `cumulative_reward` | `float` | Running reward total |
| `reward` | `float` | Step reward |
| `done` | `bool` | Episode complete |

---

## Reward Design

Dense reward every step:

| Event | Reward |
|---|---|
| On-time delivery | +2.0 |
| Late delivery | +0.8 - 0.03 × late_minutes |
| Order assigned (SLA slack bonus) | +0.02 × min(15, slack) |
| Order rejected | -0.25 |
| Order auto-cancelled (overdue) | -1.2 |
| Invalid action | -0.5 |
| Idle courier | -0.003 per courier |
| Reposition movement | -0.02 × travel_time |
| Wait with pending orders | -0.01 × pending_count |

---

## Grader

`/grader` returns score in `[0.0, 1.0]` based on:

```
score = 0.35 × on_time_rate
      + 0.20 × delivery_speed_term
      + 0.20 × courier_utilization
      + 0.10 × fairness_score
      + 0.15 × (1 - cancellation_penalty)
```

Where:
- **on_time_rate** = on_time / total_delivered
- **delivery_speed_term** = max(0, 1 - avg_delivery_minutes / 60)
- **courier_utilization** = delivered / (total_created - rejected)
- **fairness_score** = 1 - (max_load - min_load) / max_load across couriers
- **cancellation_penalty** = cancellation_rate + 0.5 × rejection_rate

---

## Baseline Results

| Task | Policy | Score | On-Time | Cancel | Avg Delivery |
|---|---|---|---|---|---|
| easy | nearest | 0.858 | 98.9% | 0.0% | 18.2 min |
| easy | hybrid | 0.863 | 100.0% | 0.0% | 18.0 min |
| easy | **ddqn_per_v1** | **0.905** | 98.8% | 0.0% | 18.5 min |
| medium | hybrid | 0.807 | 92.9% | 1.1% | 23.1 min |
| medium | **ddqn_per_v1** | **0.864** | 93.3% | 0.0% | 22.2 min |
| hard | **hybrid** | **0.698** | 71.1% | 7.2% | 25.8 min |
| hard | ddqn_per_v1 | 0.639 | 57.1% | 2.4% | 30.1 min |

DDQN+PER outperforms heuristics on easy/medium. Hybrid heuristic is strongest on hard.

---

## Quick Start

```bash
cd food_delivery_env_v2
uv sync
uv run python -m server.app
```

In another terminal:

```bash
uv run python scripts/run_baseline.py --url http://localhost:8000 --episodes 3
```

### Development Dependencies

```bash
uv sync --dev           # Installs pytest and other dev tooling
```

### Run Tests

```bash
uv run pytest tests/ -v
```

---

## Using the Client

```python
from food_delivery_env_v2 import FoodDeliveryEnv, FoodDeliveryAction

with FoodDeliveryEnv(base_url="http://localhost:8000") as env:
    obs = env.reset(task="easy")
    print(f"Pending orders: {len(obs.pending_orders)}")

    action = FoodDeliveryAction(
        action_type="assign",
        order_id=obs.pending_orders[0].order_id,
        courier_id=obs.couriers[0].courier_id,
    )
    result = env.step(action)
    print(f"Reward: {result.reward}")
```

---

## LLM Inference

```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
export HF_TOKEN="<your_key>"
export BENCHMARK_URL="http://localhost:8000"
export TASK_NAME="medium"

uv run python inference.py
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Reset environment (optional: `{"task": "easy"}`) |
| `/step` | POST | Execute action |
| `/state` | GET | Get current episode state |
| `/tasks` | GET | List tasks and action schema |
| `/grader` | POST | Run grader evaluation |
| `/baseline` | POST | Run baseline across all tasks |
| `/evaluate` | POST | Evaluate any policy |
| `/policies` | GET | List available policies |
| `/health` | GET | Health check |

---

## Training Pipelines

### DDQN + Prioritized Experience Replay

```bash
uv run python -m training.train_ddqn_per --episodes 300 --policy-id ddqn_per_v1
```

Meta-action DDQN with prioritized replay over dispatch heuristics. Uses action masking to ensure only valid actions are selected.

### Masked PPO

```bash
uv run python -m training.train_ppo_masked --updates 120 --policy-id ppo_masked_v1
```

Proximal Policy Optimization with action masking over high-level dispatch templates.

---

## Deployment

### Docker

```bash
docker build -t food-delivery-dispatch:latest .
docker run --rm -p 8000:8000 food-delivery-dispatch:latest
curl http://localhost:8000/health
```

### Hugging Face Spaces

```bash
openenv push
```

The deployed space includes:
- **Web Interface** at `/web`
- **API Docs** at `/docs`
- **Health Check** at `/health`
- **WebSocket** at `/ws`

---

## Project Structure

```
food_delivery_env_v2/
├── __init__.py                 # Package exports
├── client.py                   # FoodDeliveryEnv client
├── models.py                   # Action/Observation models
├── decision.py                 # Heuristic policies
├── inference.py                # LLM inference script
├── scripts/
│   └── run_baseline.py         # Baseline evaluation
├── validate-submission.sh      # Pre-submission validator
├── openenv.yaml                # OpenEnv manifest
├── pyproject.toml              # Package config
├── Dockerfile                  # Container definition
├── server/
│   ├── __init__.py
│   ├── app.py                  # FastAPI application
│   ├── food_delivery_environment.py  # Core simulation
│   ├── grader.py               # Multi-metric grader
│   └── Dockerfile
├── training/
│   ├── __init__.py
│   ├── common.py               # Shared utilities
│   ├── inference.py            # Trained policy inference
│   ├── train_ddqn_per.py       # DDQN+PER training
│   ├── train_ppo_masked.py     # Masked PPO training
│   └── models/
│       ├── ddqn_per_v1.pt      # Trained DDQN checkpoint
│       ├── ppo_masked_v1.pt    # Trained PPO checkpoint
│       └── registry.json       # Policy registry
└── tests/
    ├── conftest.py
    ├── test_api_endpoints.py
    └── test_grader_and_env.py
```
