from __future__ import annotations

from dataclasses import dataclass

from decision import choose_heuristic, choose_meta_action
from server.food_delivery_environment import FoodDeliveryDispatchEnvironment, SCENARIOS
from models import FoodDeliveryObservation
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


def _choose_action(policy_id: str, obs: FoodDeliveryObservation):
    if policy_id in {"nearest", "deadline", "hybrid"}:
        return choose_heuristic(policy_id, obs)
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

    if (
        resolved_policy_id not in {"nearest", "deadline", "hybrid"}
        and get_policy_record(resolved_policy_id) is None
    ):
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

        while not obs.done:
            action = _choose_action(resolved_policy_id, obs)
            obs = env.step(action)

        total_created = max(1, obs.total_orders_created)
        on_time_rate = obs.total_on_time / max(1, obs.total_delivered)
        cancellation_rate = obs.total_cancelled / total_created
        rejection_rate = obs.total_rejected / total_created

        # utilization proxy from busy time in visible couriers
        # at terminal observation, busy_for is near zero, so we use delivered load ratio proxy
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

    # Grader score in [0,1]
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
