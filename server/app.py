from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from openenv.core.env_server.http_server import create_app

from models import FoodDeliveryAction, FoodDeliveryObservation
from server.food_delivery_environment import FoodDeliveryDispatchEnvironment, SCENARIOS
from server.grader import run_policy_evaluation
from server.web_ui import mount
from training.inference import list_registered_policies


app = create_app(
    FoodDeliveryDispatchEnvironment,
    FoodDeliveryAction,
    FoodDeliveryObservation,
    env_name="food_delivery_dispatch",
    max_concurrent_envs=2,
)

router = APIRouter()


TASKS = [
    {
        "task_id": "easy",
        "name": "Easy - Stable Demand",
        "description": "Moderate order flow, relaxed SLA, low traffic volatility.",
    },
    {
        "task_id": "medium",
        "name": "Medium - Peak Variability",
        "description": "Higher demand bursts, tighter SLA, moderate traffic spikes.",
    },
    {
        "task_id": "hard",
        "name": "Hard - Adversarial Peak",
        "description": "High demand volatility, strict SLA, severe traffic multipliers.",
    },
]


POLICIES = [
    {
        "policy_id": "nearest",
        "name": "Nearest/Least Busy Heuristic",
        "family": "heuristic",
    },
    {
        "policy_id": "deadline",
        "name": "Deadline Aware Heuristic",
        "family": "heuristic",
    },
    {
        "policy_id": "hybrid",
        "name": "Hybrid Dispatch + Rejection Heuristic",
        "family": "heuristic",
    },
    {
        "policy_id": "auto_best",
        "name": "Auto Best (ddqn for easy/medium, hybrid for hard)",
        "family": "composite",
    },
]


class GraderRequest(BaseModel):
    task_id: str = Field(default="medium")
    policy_id: str = Field(default="hybrid")
    episodes: int = Field(default=3, ge=1, le=20)


class BaselineRequest(BaseModel):
    episodes: int = Field(default=3, ge=1, le=20)


class EvaluateRequest(BaseModel):
    task_id: str = Field(default="hard")
    policy_id: str = Field(default="hybrid")
    episodes: int = Field(default=5, ge=1, le=50)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "env": "food_delivery_dispatch", "version": "2.0"}


@app.get("/tasks")
async def get_tasks() -> dict:
    return {
        "tasks": TASKS,
        "action_schema": {
            "action_type": ["assign", "reject", "reposition", "wait"],
            "assign": {"required": ["order_id", "courier_id"]},
            "reject": {"required": ["order_id"]},
            "reposition": {"required": ["courier_id", "target_zone"]},
            "wait": {"required": []},
        },
        "available_task_ids": list(SCENARIOS.keys()),
    }


@app.get("/policies")
async def get_policies() -> dict:
    trained = [
        {
            "policy_id": p.get("policy_id"),
            "name": p.get("policy_id"),
            "family": p.get("algo", "trained"),
            "created_at": p.get("created_at"),
        }
        for p in list_registered_policies()
    ]
    return {"heuristic_policies": POLICIES, "trained_policies": trained}


@app.post("/grader")
async def grader(request: GraderRequest) -> dict:
    metrics = run_policy_evaluation(
        task_id=request.task_id,
        policy_id=request.policy_id,
        episodes=request.episodes,
    )
    return {
        "task_id": metrics.task_id,
        "policy_id": metrics.policy_id,
        "episodes": metrics.episodes,
        "score": metrics.score,
        "metrics": {
            "on_time_rate": metrics.on_time_rate,
            "cancellation_rate": metrics.cancellation_rate,
            "rejection_rate": metrics.rejection_rate,
            "avg_delivery_minutes": metrics.avg_delivery_minutes,
            "courier_utilization": metrics.courier_utilization,
            "fairness_score": metrics.fairness_score,
        },
    }


@app.post("/baseline")
async def baseline(request: BaselineRequest) -> dict:
    policies = ["nearest", "deadline", "hybrid", "ddqn_per_v1", "auto_best"]

    rows = []
    for task in SCENARIOS:
        for policy in policies:
            m = run_policy_evaluation(
                task_id=task, policy_id=policy, episodes=request.episodes
            )
            rows.append(
                {
                    "task_id": task,
                    "policy_id": policy,
                    "score": m.score,
                    "on_time_rate": m.on_time_rate,
                    "avg_delivery_minutes": m.avg_delivery_minutes,
                    "cancellation_rate": m.cancellation_rate,
                    "rejection_rate": m.rejection_rate,
                }
            )
    return {
        "episodes": request.episodes,
        "policies": policies,
        "results": rows,
    }


@app.get("/baseline")
async def baseline_get(episodes: int = 3) -> dict:
    request = BaselineRequest(episodes=episodes)
    return await baseline(request)


@app.post("/evaluate")
async def evaluate(request: EvaluateRequest) -> dict:
    m = run_policy_evaluation(
        task_id=request.task_id,
        policy_id=request.policy_id,
        episodes=request.episodes,
    )
    return {
        "task_id": m.task_id,
        "policy_id": m.policy_id,
        "episodes": m.episodes,
        "score": m.score,
        "metrics": {
            "on_time_rate": m.on_time_rate,
            "cancellation_rate": m.cancellation_rate,
            "rejection_rate": m.rejection_rate,
            "avg_delivery_minutes": m.avg_delivery_minutes,
            "courier_utilization": m.courier_utilization,
            "fairness_score": m.fairness_score,
        },
    }


app.include_router(router)
mount(app)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
