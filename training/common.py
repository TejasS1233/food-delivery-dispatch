from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json

from models import FoodDeliveryObservation


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
REGISTRY_PATH = MODELS_DIR / "registry.json"


@dataclass
class PolicyRecord:
    policy_id: str
    algo: str
    checkpoint_path: str
    task_mix: list[str]
    created_at: str
    notes: str


def obs_to_vector(
    obs: FoodDeliveryObservation, top_k_orders: int = 3, top_k_couriers: int = 3
) -> list[float]:
    v: list[float] = []

    horizon = max(1, obs.horizon)
    v.append(obs.minute / horizon)
    v.append(min(1.0, len(obs.pending_orders) / 20.0))
    v.append(min(1.0, len(obs.couriers) / 40.0))
    v.append(min(1.0, obs.total_delivered / max(1, obs.total_orders_created)))
    v.append(min(1.0, obs.total_rejected / max(1, obs.total_orders_created)))
    v.append(min(1.0, obs.total_cancelled / max(1, obs.total_orders_created)))
    v.append(min(1.0, obs.average_delivery_minutes / 60.0))

    orders = sorted(
        obs.pending_orders, key=lambda o: (o.sla_remaining_minutes, -o.age_minutes)
    )
    for idx in range(top_k_orders):
        if idx < len(orders):
            o = orders[idx]
            v.extend(
                [
                    min(1.0, o.age_minutes / 60.0),
                    min(1.0, o.prep_remaining_minutes / 20.0),
                    max(-1.0, min(1.0, o.sla_remaining_minutes / 60.0)),
                    min(1.0, o.estimated_best_eta / 60.0),
                ]
            )
        else:
            v.extend([0.0, 0.0, 0.0, 0.0])

    couriers = sorted(obs.couriers, key=lambda c: (c.queue_length, c.busy_for_minutes))
    for idx in range(top_k_couriers):
        if idx < len(couriers):
            c = couriers[idx]
            v.extend(
                [
                    min(1.0, c.busy_for_minutes / 60.0),
                    min(1.0, c.queue_length / 4.0),
                    min(1.0, c.reposition_remaining_minutes / 30.0),
                ]
            )
        else:
            v.extend([0.0, 0.0, 0.0])

    return v


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {"policies": []}
    with REGISTRY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry: dict):
    with REGISTRY_PATH.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def register_policy(
    policy_id: str, algo: str, checkpoint_path: Path, task_mix: list[str], notes: str
):
    registry = load_registry()
    existing = [
        p for p in registry.get("policies", []) if p.get("policy_id") != policy_id
    ]
    existing.append(
        PolicyRecord(
            policy_id=policy_id,
            algo=algo,
            checkpoint_path=str(checkpoint_path),
            task_mix=task_mix,
            created_at=datetime.utcnow().isoformat() + "Z",
            notes=notes,
        ).__dict__
    )
    registry["policies"] = existing
    save_registry(registry)
