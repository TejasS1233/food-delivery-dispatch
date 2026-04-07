"""
Inference script for Food Delivery Dispatch OpenEnv.

Required environment variables for LLM:
- API_BASE_URL: OpenAI-compatible LLM endpoint (Groq works with OpenAI client)
- MODEL_NAME: model identifier
- HF_TOKEN: API key (or GROQ_API_KEY / API_KEY fallback)

Optional:
- LOCAL_IMAGE_NAME: reserved for docker-based env startup flows

This script emits exactly these stdout line types:
- [START] ...
- [STEP]  ... (one per step)
- [END]   ... (always)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import requests
from openai import OpenAI


LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")
API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

BENCHMARK_URL = os.getenv("BENCHMARK_URL", "http://localhost:8000")
TASK_NAME = os.getenv("TASK_NAME", "medium")
BENCHMARK = os.getenv("BENCHMARK", "food_delivery_dispatch")
MAX_STEPS = int(os.getenv("MAX_STEPS", "40"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "220"))


SYSTEM_PROMPT = (
    "You are a food-delivery dispatch agent. "
    "Return exactly one JSON object with keys: "
    "action_type (assign|reject|reposition|wait), order_id (optional), "
    "courier_id (optional), target_zone (optional int). "
    "Prefer assign for feasible urgent orders, reject impossible ones, and reposition only when useful."
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: str | None
) -> None:
    err = _single_line(error) if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    reward_csv = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={reward_csv}",
        flush=True,
    )


class HttpEnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def reset(self, task: str) -> dict[str, Any]:
        resp = self.session.post(
            f"{self.base_url}/reset", json={"task": task}, timeout=30
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        resp = self.session.post(
            f"{self.base_url}/step", json={"action": action}, timeout=30
        )
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self.session.close()


def extract_result_fields(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], float, bool, str | None]:
    observation = payload.get("observation", payload)
    reward = payload.get("reward", observation.get("reward", 0.0))
    done = bool(payload.get("done", observation.get("done", False)))
    last_error = payload.get("last_action_error")
    return observation, float(reward or 0.0), done, last_error


def action_to_str(action: dict[str, Any]) -> str:
    return _single_line(json.dumps(action, separators=(",", ":")))


def _single_line(text: str | None) -> str:
    return (text or "").replace("\n", " ").replace("\r", " ").strip()


def build_user_prompt(step: int, obs: dict[str, Any], rewards: list[float]) -> str:
    pending = obs.get("pending_orders", [])[:4]
    couriers = obs.get("couriers", [])[:4]
    summary = {
        "step": step,
        "task": obs.get("task_id"),
        "minute": obs.get("minute"),
        "horizon": obs.get("horizon"),
        "pending_count": len(obs.get("pending_orders", [])),
        "total_delivered": obs.get("total_delivered"),
        "total_rejected": obs.get("total_rejected"),
        "total_cancelled": obs.get("total_cancelled"),
        "avg_delivery_minutes": obs.get("average_delivery_minutes"),
        "recent_rewards": [round(r, 2) for r in rewards[-5:]],
        "pending_orders": pending,
        "couriers": couriers,
    }
    return "Choose one next action as JSON only.\n" + json.dumps(
        summary, ensure_ascii=True
    )


def parse_action(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {"action_type": "wait"}
    try:
        obj = json.loads(match.group(0))
        action_type = obj.get("action_type", "wait")
        if action_type not in {"assign", "reject", "reposition", "wait"}:
            return {"action_type": "wait"}

        action: dict[str, Any] = {"action_type": action_type}
        if obj.get("order_id") is not None:
            action["order_id"] = str(obj["order_id"])
        if obj.get("courier_id") is not None:
            action["courier_id"] = str(obj["courier_id"])
        if obj.get("target_zone") is not None:
            try:
                action["target_zone"] = int(obj["target_zone"])
            except Exception:
                pass
        return action
    except Exception:
        return {"action_type": "wait"}


def choose_action_with_llm(
    client: OpenAI, step: int, obs: dict[str, Any], rewards: list[float]
) -> dict[str, Any]:
    prompt = build_user_prompt(step, obs, rewards)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        content = (completion.choices[0].message.content or "").strip()
        return parse_action(content)
    except Exception:
        return choose_fallback_action(obs)


def choose_fallback_action(obs: dict[str, Any]) -> dict[str, Any]:
    pending = obs.get("pending_orders", [])
    couriers = obs.get("couriers", [])
    if not pending:
        return {"action_type": "wait"}

    order = sorted(
        pending,
        key=lambda o: (o.get("sla_remaining_minutes", 999), -o.get("age_minutes", 0)),
    )[0]

    best_eta = int(order.get("estimated_best_eta", 999))
    sla_left = int(order.get("sla_remaining_minutes", 999))
    if best_eta > sla_left + 5:
        return {"action_type": "reject", "order_id": str(order.get("order_id", ""))}

    available = sorted(
        couriers,
        key=lambda c: (c.get("queue_length", 99), c.get("busy_for_minutes", 999)),
    )
    if not available:
        return {"action_type": "wait"}
    return {
        "action_type": "assign",
        "order_id": str(order.get("order_id", "")),
        "courier_id": str(available[0].get("courier_id", "")),
    }


def compute_score(obs: dict[str, Any]) -> float:
    total_created = max(1, int(obs.get("total_orders_created", 0) or 0))
    total_delivered = max(1, int(obs.get("total_delivered", 0) or 0))
    total_on_time = int(obs.get("total_on_time", 0) or 0)
    total_cancelled = int(obs.get("total_cancelled", 0) or 0)
    total_rejected = int(obs.get("total_rejected", 0) or 0)
    avg_delivery = float(obs.get("average_delivery_minutes", 0.0) or 0.0)

    on_time_rate = total_on_time / total_delivered
    cancellation_rate = total_cancelled / total_created
    rejection_rate = total_rejected / total_created
    delivery_term = max(0.0, 1.0 - (avg_delivery / 60.0))

    score = (
        0.45 * on_time_rate
        + 0.25 * delivery_term
        + 0.30 * (1.0 - min(1.0, cancellation_rate + 0.5 * rejection_rate))
    )
    return max(0.0, min(1.0, score))


def main() -> None:
    if os.getenv("HF_TOKEN") is None:
        raise ValueError("HF_TOKEN environment variable is required")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = HttpEnvClient(BENCHMARK_URL)

    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_payload = env.reset(TASK_NAME)
        obs, _, done, _ = extract_result_fields(reset_payload)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = choose_action_with_llm(client, step, obs, rewards)

            try:
                step_payload = env.step(action)
                obs, reward, done, last_error = extract_result_fields(step_payload)
            except Exception as exc:
                reward = 0.0
                done = True
                last_error = _single_line(str(exc))

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_to_str(action),
                reward=reward,
                done=done,
                error=last_error,
            )

            if done:
                break

        score = compute_score(obs)
        success = score >= 0.5
    except Exception:
        success = False
    finally:
        env.close()
        log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    main()
