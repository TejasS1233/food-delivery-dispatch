from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from decision import choose_heuristic, choose_meta_action
from server.food_delivery_environment import FoodDeliveryDispatchEnvironment, SCENARIOS
from models import FoodDeliveryAction, FoodDeliveryObservation
from training.inference import get_policy_record, predict_meta_action


@dataclass
class EvaluationMetrics:
    task_id: str
    policy_id: str
    episodes: int
    score: float
    on_time_rate: float
    cancellation_rate: float
    rejection_rate: float
    avg_delivery_minutes: float
    courier_utilization: float
    fairness_score: float


SYSTEM_PROMPT = (
    "You are a food-delivery dispatch agent. "
    "Return exactly one JSON object with keys: "
    "action_type (assign|reject|reposition|wait), order_id (optional), "
    "courier_id (optional), target_zone (optional int). "
    "Prefer assign for feasible urgent orders, reject impossible ones, and reposition only when useful."
)

LLM_API_KEY = os.getenv("HF_TOKEN") or os.getenv("GROQ_API_KEY") or os.getenv("API_KEY")
LLM_API_BASE = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")


def _parse_llm_action(text: str) -> FoodDeliveryAction:
    text = (text or "").strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return FoodDeliveryAction(action_type="wait")
    try:
        obj = json.loads(match.group(0))
        action_type = obj.get("action_type", "wait")
        if action_type not in {"assign", "reject", "reposition", "wait"}:
            return FoodDeliveryAction(action_type="wait")
        action = FoodDeliveryAction(action_type=action_type)
        if obj.get("order_id") is not None:
            action.order_id = str(obj["order_id"])
        if obj.get("courier_id") is not None:
            action.courier_id = str(obj["courier_id"])
        if obj.get("target_zone") is not None:
            try:
                action.target_zone = int(obj["target_zone"])
            except Exception:
                pass
        return action
    except Exception:
        return FoodDeliveryAction(action_type="wait")


def _llm_fallback_action(obs: FoodDeliveryObservation) -> FoodDeliveryAction:
    if not obs.pending_orders:
        return FoodDeliveryAction(action_type="wait")
    order = sorted(
        obs.pending_orders,
        key=lambda o: (o.sla_remaining_minutes, -o.age_minutes),
    )[0]
    if order.estimated_best_eta > order.sla_remaining_minutes + 5:
        return FoodDeliveryAction(action_type="reject", order_id=order.order_id)
    available = sorted(
        obs.couriers,
        key=lambda c: (c.queue_length, c.busy_for_minutes),
    )
    if not available:
        return FoodDeliveryAction(action_type="wait")
    return FoodDeliveryAction(
        action_type="assign",
        order_id=order.order_id,
        courier_id=available[0].courier_id,
    )


def _choose_action_with_llm(
    obs: FoodDeliveryObservation, step: int, rewards: list[float]
) -> FoodDeliveryAction:
    try:
        from openai import OpenAI
    except ImportError:
        return _llm_fallback_action(obs)

    if not LLM_API_KEY:
        return _llm_fallback_action(obs)

    pending = obs.pending_orders[:4]
    couriers = obs.couriers[:4]
    summary = {
        "step": step,
        "task": obs.task_id,
        "minute": obs.minute,
        "horizon": obs.horizon,
        "pending_count": len(obs.pending_orders),
        "total_delivered": obs.total_delivered,
        "total_rejected": obs.total_rejected,
        "total_cancelled": obs.total_cancelled,
        "avg_delivery_minutes": obs.average_delivery_minutes,
        "recent_rewards": [round(r, 2) for r in rewards[-5:]],
        "pending_orders": [o.model_dump() for o in pending],
        "couriers": [c.model_dump() for c in couriers],
    }
    prompt = "Choose one next action as JSON only.\n" + json.dumps(
        summary, ensure_ascii=True
    )

    try:
        client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=220,
            stream=False,
        )
        content = (completion.choices[0].message.content or "").strip()
        return _parse_llm_action(content)
    except Exception:
        return _llm_fallback_action(obs)


def _choose_action(
    policy_id: str,
    obs: FoodDeliveryObservation,
    step: int = 0,
    rewards: list[float] | None = None,
):
    if rewards is None:
        rewards = []
    if policy_id in {"nearest", "deadline", "hybrid"}:
        return choose_heuristic(policy_id, obs)
    if policy_id == "llm":
        return _choose_action_with_llm(obs, step, rewards)
    action_id = predict_meta_action(policy_id, obs)
    return choose_meta_action(action_id, obs).action


def _resolve_policy_id(task_id: str, policy_id: str) -> str:
    if policy_id != "auto_best":
        return policy_id

    if task_id == "hard":
        return "hybrid"

    if get_policy_record("ddqn_per_v1") is not None:
        return "ddqn_per_v1"
    return "hybrid"


def run_policy_evaluation(
    task_id: str, policy_id: str, episodes: int = 3
) -> EvaluationMetrics:
    if task_id not in SCENARIOS:
        raise ValueError(f"Unknown task: {task_id}")
    resolved_policy_id = _resolve_policy_id(task_id, policy_id)

    is_llm = policy_id == "llm"
    is_heuristic = resolved_policy_id in {"nearest", "deadline", "hybrid"}
    is_trained = get_policy_record(resolved_policy_id) is not None

    if not is_llm and not is_heuristic and not is_trained:
        raise ValueError(f"Unknown policy: {policy_id}")

    on_time_rate_sum = 0.0
    cancellation_rate_sum = 0.0
    rejection_rate_sum = 0.0
    avg_delivery_sum = 0.0
    utilization_sum = 0.0
    fairness_sum = 0.0

    for _ in range(episodes):
        env = FoodDeliveryDispatchEnvironment(task=task_id)
        obs = env.reset(task=task_id)
        rewards: list[float] = []
        step = 0

        while not obs.done:
            action = _choose_action(resolved_policy_id, obs, step, rewards)
            obs = env.step(action)
            rewards.append(obs.reward)
            step += 1

        total_created = max(1, obs.total_orders_created)
        on_time_rate = obs.total_on_time / max(1, obs.total_delivered)
        cancellation_rate = obs.total_cancelled / total_created
        rejection_rate = obs.total_rejected / total_created

        utilization_proxy = min(
            1.0, obs.total_delivered / max(1, total_created - obs.total_rejected)
        )

        loads = [c.queue_length for c in obs.couriers]
        fairness = 1.0
        if loads:
            max_load = max(loads)
            min_load = min(loads)
            fairness = (
                1.0 if max_load == 0 else 1.0 - ((max_load - min_load) / max_load)
            )

        on_time_rate_sum += on_time_rate
        cancellation_rate_sum += cancellation_rate
        rejection_rate_sum += rejection_rate
        avg_delivery_sum += obs.average_delivery_minutes
        utilization_sum += utilization_proxy
        fairness_sum += max(0.0, fairness)

    on_time_rate = on_time_rate_sum / episodes
    cancellation_rate = cancellation_rate_sum / episodes
    rejection_rate = rejection_rate_sum / episodes
    avg_delivery = avg_delivery_sum / episodes
    utilization = utilization_sum / episodes
    fairness = fairness_sum / episodes

    delivery_term = max(0.0, 1.0 - (avg_delivery / 60.0))
    score = (
        0.35 * on_time_rate
        + 0.20 * delivery_term
        + 0.20 * utilization
        + 0.10 * fairness
        + 0.15 * (1.0 - min(1.0, cancellation_rate + 0.5 * rejection_rate))
    )
    score = max(0.0, min(1.0, score))

    return EvaluationMetrics(
        task_id=task_id,
        policy_id=resolved_policy_id,
        episodes=episodes,
        score=round(score, 4),
        on_time_rate=round(on_time_rate, 4),
        cancellation_rate=round(cancellation_rate, 4),
        rejection_rate=round(rejection_rate, 4),
        avg_delivery_minutes=round(avg_delivery, 2),
        courier_utilization=round(utilization, 4),
        fairness_score=round(fairness, 4),
    )
