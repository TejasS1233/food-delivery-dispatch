from __future__ import annotations

from dataclasses import dataclass

from models import FoodDeliveryAction, FoodDeliveryObservation


META_ACTIONS = [
    "nearest",
    "deadline",
    "hybrid",
    "reject_worst",
    "reposition_hotspot",
]


@dataclass
class ActionChoice:
    action: FoodDeliveryAction
    label: str


def action_mask(obs: FoodDeliveryObservation) -> list[int]:
    has_pending = len(obs.pending_orders) > 0
    has_idle = any(
        c.queue_length == 0 and c.reposition_remaining_minutes == 0
        for c in obs.couriers
    )
    # nearest, deadline, hybrid are always valid (fall back to wait when no orders)
    # reject_worst needs at least one pending order
    # reposition_hotspot needs at least one idle courier
    return [1, 1, 1, 1 if has_pending else 0, 1 if has_idle else 0]


def choose_heuristic(
    policy_id: str, obs: FoodDeliveryObservation
) -> FoodDeliveryAction:
    if not obs.pending_orders:
        if policy_id == "hybrid":
            idle = [
                c
                for c in obs.couriers
                if c.queue_length == 0 and c.reposition_remaining_minutes == 0
            ]
            if idle:
                return FoodDeliveryAction(
                    action_type="reposition",
                    courier_id=idle[0].courier_id,
                    target_zone=idle[0].zone,
                )
        return FoodDeliveryAction(action_type="wait")

    if policy_id == "nearest":
        order = sorted(
            obs.pending_orders, key=lambda o: (o.sla_remaining_minutes, -o.age_minutes)
        )[0]
        courier = sorted(
            obs.couriers, key=lambda c: (c.queue_length, c.busy_for_minutes)
        )[0]
        return FoodDeliveryAction(
            action_type="assign",
            order_id=order.order_id,
            courier_id=courier.courier_id,
        )

    if policy_id == "deadline":
        order = sorted(
            obs.pending_orders, key=lambda o: (o.sla_remaining_minutes, -o.age_minutes)
        )[0]
        if order.sla_remaining_minutes <= 6:
            return FoodDeliveryAction(action_type="reject", order_id=order.order_id)
        courier = sorted(
            obs.couriers, key=lambda c: (c.busy_for_minutes, c.queue_length)
        )[0]
        return FoodDeliveryAction(
            action_type="assign",
            order_id=order.order_id,
            courier_id=courier.courier_id,
        )

    # hybrid
    order = sorted(
        obs.pending_orders,
        key=lambda o: (o.estimated_best_eta - o.sla_remaining_minutes, -o.age_minutes),
    )[0]
    if order.estimated_best_eta > order.sla_remaining_minutes + 5:
        return FoodDeliveryAction(action_type="reject", order_id=order.order_id)
    courier = sorted(obs.couriers, key=lambda c: (c.queue_length, c.busy_for_minutes))[
        0
    ]
    return FoodDeliveryAction(
        action_type="assign",
        order_id=order.order_id,
        courier_id=courier.courier_id,
    )


def choose_meta_action(action_id: int, obs: FoodDeliveryObservation) -> ActionChoice:
    action_id = max(0, min(action_id, len(META_ACTIONS) - 1))
    label = META_ACTIONS[action_id]

    if label in {"nearest", "deadline", "hybrid"}:
        return ActionChoice(action=choose_heuristic(label, obs), label=label)

    if label == "reject_worst":
        if not obs.pending_orders:
            return ActionChoice(
                action=FoodDeliveryAction(action_type="wait"), label="wait"
            )
        worst = sorted(
            obs.pending_orders,
            key=lambda o: (
                o.sla_remaining_minutes - o.estimated_best_eta,
                -o.age_minutes,
            ),
        )[0]
        return ActionChoice(
            action=FoodDeliveryAction(action_type="reject", order_id=worst.order_id),
            label=label,
        )

    # reposition_hotspot
    idle = [
        c
        for c in obs.couriers
        if c.queue_length == 0 and c.reposition_remaining_minutes == 0
    ]
    if not idle:
        return ActionChoice(action=FoodDeliveryAction(action_type="wait"), label="wait")
    if obs.pending_orders:
        # reposition toward most urgent pending restaurant to reduce pickup delay
        target = sorted(obs.pending_orders, key=lambda o: o.sla_remaining_minutes)[
            0
        ].restaurant_zone
    else:
        target = idle[0].zone
    return ActionChoice(
        action=FoodDeliveryAction(
            action_type="reposition",
            courier_id=idle[0].courier_id,
            target_zone=target,
        ),
        label=label,
    )
