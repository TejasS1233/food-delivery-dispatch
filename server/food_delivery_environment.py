from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Optional
from uuid import uuid4
import math
import random

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import (
    CourierView,
    FoodDeliveryAction,
    FoodDeliveryObservation,
    PendingOrderView,
)


@dataclass
class ScenarioConfig:
    task_id: str
    width: int
    height: int
    horizon: int
    courier_count: int
    max_pending_orders_view: int
    base_seed: int
    mean_prep: float
    std_prep: float
    sla_minutes: int
    cancel_after_minutes: int
    max_queue_per_courier: int
    lunch_peak_multiplier: float
    dinner_peak_multiplier: float
    traffic_peak_multiplier: float
    base_arrival_rate: float
    burst_probability: float


SCENARIOS: dict[str, ScenarioConfig] = {
    "easy": ScenarioConfig(
        task_id="easy",
        width=12,
        height=12,
        horizon=180,
        courier_count=12,
        max_pending_orders_view=12,
        base_seed=101,
        mean_prep=8.0,
        std_prep=2.0,
        sla_minutes=38,
        cancel_after_minutes=22,
        max_queue_per_courier=2,
        lunch_peak_multiplier=1.20,
        dinner_peak_multiplier=1.25,
        traffic_peak_multiplier=1.10,
        base_arrival_rate=0.42,
        burst_probability=0.01,
    ),
    "medium": ScenarioConfig(
        task_id="medium",
        width=16,
        height=16,
        horizon=240,
        courier_count=14,
        max_pending_orders_view=14,
        base_seed=202,
        mean_prep=9.0,
        std_prep=3.0,
        sla_minutes=34,
        cancel_after_minutes=18,
        max_queue_per_courier=2,
        lunch_peak_multiplier=1.35,
        dinner_peak_multiplier=1.55,
        traffic_peak_multiplier=1.25,
        base_arrival_rate=0.55,
        burst_probability=0.03,
    ),
    "hard": ScenarioConfig(
        task_id="hard",
        width=20,
        height=20,
        horizon=300,
        courier_count=14,
        max_pending_orders_view=16,
        base_seed=303,
        mean_prep=10.0,
        std_prep=3.5,
        sla_minutes=30,
        cancel_after_minutes=14,
        max_queue_per_courier=3,
        lunch_peak_multiplier=1.60,
        dinner_peak_multiplier=1.95,
        traffic_peak_multiplier=1.45,
        base_arrival_rate=0.68,
        burst_probability=0.08,
    ),
}


@dataclass
class Order:
    order_id: str
    created_minute: int
    restaurant_zone: int
    customer_zone: int
    prep_ready_minute: int
    deadline_minute: int
    status: str = "pending"
    assigned_courier_id: str | None = None
    assigned_minute: int | None = None
    delivered_minute: int | None = None
    estimated_eta: int | None = None


@dataclass
class DeliveryLeg:
    order_id: str
    completion_minute: int
    customer_zone: int


@dataclass
class Courier:
    courier_id: str
    zone: int
    queue: list[DeliveryLeg] = field(default_factory=list)
    reposition_target_zone: int | None = None
    reposition_finish_minute: int = 0
    active_minutes: int = 0
    idle_minutes: int = 0


class FoodDeliveryDispatchEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task: str = "easy"):
        self.task = task if task in SCENARIOS else "easy"
        self.config = SCENARIOS[self.task]
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._rng = random.Random(self.config.base_seed)
        self._minute = 0
        self._orders: dict[str, Order] = {}
        self._pending_order_ids: list[str] = []
        self._couriers: dict[str, Courier] = {}
        self._cumulative_reward = 0.0
        self._step_reward = 0.0
        self._next_order_id = 1
        self._delivered_minutes: list[int] = []
        self._setup_episode(seed=self.config.base_seed)

    def reset(self, task: Optional[str] = None) -> FoodDeliveryObservation:
        if task and task in SCENARIOS:
            self.task = task
            self.config = SCENARIOS[task]

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._setup_episode(seed=self.config.base_seed)
        return self._build_observation(done=False)

    def step(self, action: FoodDeliveryAction) -> FoodDeliveryObservation:
        self._state.step_count += 1
        self._step_reward = 0.0

        self._apply_action(action)
        self._advance_one_minute()
        done = self._minute >= self.config.horizon
        return self._build_observation(done=done)

    @property
    def state(self) -> State:
        return self._state

    def _setup_episode(self, seed: int):
        self._rng = random.Random(seed)
        self._minute = 0
        self._orders = {}
        self._pending_order_ids = []
        self._cumulative_reward = 0.0
        self._step_reward = 0.0
        self._next_order_id = 1
        self._delivered_minutes = []
        self._couriers = {}

        for idx in range(self.config.courier_count):
            start_zone = self._zone_id(self.config.width // 2, self.config.height // 2)
            courier_id = f"C{idx + 1:02d}"
            self._couriers[courier_id] = Courier(courier_id=courier_id, zone=start_zone)

        for _ in range(5):
            self._spawn_orders()
            self._minute += 1
        self._minute = 0

    def _build_observation(self, done: bool) -> FoodDeliveryObservation:
        pending_views = self._pending_order_views()
        courier_views = self._courier_views()
        delivered = [o for o in self._orders.values() if o.status == "delivered"]
        on_time = [
            o
            for o in delivered
            if o.delivered_minute is not None
            and o.delivered_minute <= o.deadline_minute
        ]
        late = len(delivered) - len(on_time)
        rejected = len([o for o in self._orders.values() if o.status == "rejected"])
        cancelled = len([o for o in self._orders.values() if o.status == "cancelled"])

        avg_delivery = mean(self._delivered_minutes) if self._delivered_minutes else 0.0
        self._cumulative_reward += self._step_reward

        return FoodDeliveryObservation(
            task_id=self.task,
            minute=self._minute,
            horizon=self.config.horizon,
            pending_orders=pending_views,
            couriers=courier_views,
            total_orders_created=len(self._orders),
            total_delivered=len(delivered),
            total_on_time=len(on_time),
            total_late=late,
            total_rejected=rejected,
            total_cancelled=cancelled,
            average_delivery_minutes=round(avg_delivery, 2),
            cumulative_reward=round(self._cumulative_reward, 4),
            done=done,
            reward=round(self._step_reward, 4),
            metadata={
                "version": "2.0",
                "seed": self.config.base_seed,
                "pending_count": len(self._pending_order_ids),
                "traffic_multiplier": round(self._traffic_multiplier(self._minute), 3),
            },
        )

    def _pending_order_views(self) -> list[PendingOrderView]:
        pending_orders = [
            self._orders[oid]
            for oid in self._pending_order_ids
            if self._orders[oid].status == "pending"
        ]
        pending_orders.sort(
            key=lambda o: (o.deadline_minute - self._minute, o.created_minute)
        )

        views: list[PendingOrderView] = []
        for order in pending_orders[: self.config.max_pending_orders_view]:
            prep_remaining = max(0, order.prep_ready_minute - self._minute)
            sla_remaining = order.deadline_minute - self._minute
            views.append(
                PendingOrderView(
                    order_id=order.order_id,
                    restaurant_zone=order.restaurant_zone,
                    customer_zone=order.customer_zone,
                    age_minutes=self._minute - order.created_minute,
                    prep_remaining_minutes=prep_remaining,
                    sla_remaining_minutes=sla_remaining,
                    estimated_best_eta=self._best_possible_eta(order),
                )
            )
        return views

    def _courier_views(self) -> list[CourierView]:
        views: list[CourierView] = []
        for courier in self._couriers.values():
            busy_for = self._courier_busy_for(courier)
            reposition_remaining = max(
                0, courier.reposition_finish_minute - self._minute
            )
            views.append(
                CourierView(
                    courier_id=courier.courier_id,
                    zone=courier.zone,
                    busy_for_minutes=busy_for,
                    queue_length=len(courier.queue),
                    reposition_remaining_minutes=reposition_remaining,
                )
            )
        views.sort(key=lambda c: c.courier_id)
        return views

    def _apply_action(self, action: FoodDeliveryAction):
        if action.action_type == "wait":
            self._step_reward -= 0.01 * len(self._pending_order_ids)
            return

        if action.action_type == "assign":
            if not action.order_id or not action.courier_id:
                self._step_reward -= 0.5
                return
            self._assign_order(action.order_id, action.courier_id)
            return

        if action.action_type == "reject":
            if not action.order_id:
                self._step_reward -= 0.5
                return
            self._reject_order(action.order_id)
            return

        if action.action_type == "reposition":
            if not action.courier_id or action.target_zone is None:
                self._step_reward -= 0.5
                return
            self._reposition_courier(action.courier_id, action.target_zone)
            return

        self._step_reward -= 0.5

    def _assign_order(self, order_id: str, courier_id: str):
        if order_id not in self._orders or courier_id not in self._couriers:
            self._step_reward -= 0.5
            return
        order = self._orders[order_id]
        courier = self._couriers[courier_id]
        if order.status != "pending":
            self._step_reward -= 0.5
            return
        if len(courier.queue) >= self.config.max_queue_per_courier:
            self._step_reward -= 0.25
            return

        eta, completion_minute = self._estimate_completion(courier, order)
        order.status = "assigned"
        order.assigned_courier_id = courier_id
        order.assigned_minute = self._minute
        order.estimated_eta = eta

        courier.queue.append(
            DeliveryLeg(
                order_id=order_id,
                completion_minute=completion_minute,
                customer_zone=order.customer_zone,
            )
        )
        courier.queue.sort(key=lambda leg: leg.completion_minute)

        if order_id in self._pending_order_ids:
            self._pending_order_ids.remove(order_id)

        # shaping: slight urgency bonus for tight SLA fulfillment attempts
        slack = order.deadline_minute - completion_minute
        self._step_reward += 0.02 * max(0, min(15, slack))

    def _reject_order(self, order_id: str):
        if order_id not in self._orders:
            self._step_reward -= 0.5
            return
        order = self._orders[order_id]
        if order.status != "pending":
            self._step_reward -= 0.5
            return
        order.status = "rejected"
        if order_id in self._pending_order_ids:
            self._pending_order_ids.remove(order_id)
        self._step_reward -= 0.25

    def _reposition_courier(self, courier_id: str, target_zone: int):
        courier = self._couriers.get(courier_id)
        if not courier:
            self._step_reward -= 0.5
            return
        if courier.queue:
            self._step_reward -= 0.2
            return
        target_zone = max(
            0, min(self.config.width * self.config.height - 1, target_zone)
        )
        distance = self._distance(courier.zone, target_zone)
        travel = max(
            1, int(math.ceil(distance * self._traffic_multiplier(self._minute)))
        )
        courier.reposition_target_zone = target_zone
        courier.reposition_finish_minute = self._minute + travel
        self._step_reward -= 0.02 * travel

    def _advance_one_minute(self):
        # complete deliveries
        for courier in self._couriers.values():
            completed: list[DeliveryLeg] = [
                leg for leg in courier.queue if leg.completion_minute <= self._minute
            ]
            for leg in completed:
                order = self._orders[leg.order_id]
                order.status = "delivered"
                order.delivered_minute = self._minute
                courier.zone = leg.customer_zone
                delivery_duration = order.delivered_minute - order.created_minute
                self._delivered_minutes.append(delivery_duration)

                late_minutes = max(0, order.delivered_minute - order.deadline_minute)
                if late_minutes == 0:
                    self._step_reward += 2.0
                else:
                    self._step_reward += 0.8
                    self._step_reward -= 0.03 * late_minutes
            if completed:
                courier.queue = [
                    leg for leg in courier.queue if leg.completion_minute > self._minute
                ]

        # finish repositioning
        for courier in self._couriers.values():
            if (
                courier.reposition_target_zone is not None
                and courier.reposition_finish_minute <= self._minute
                and not courier.queue
            ):
                courier.zone = courier.reposition_target_zone
                courier.reposition_target_zone = None

        # cancel overdue pending orders
        kept_pending: list[str] = []
        for order_id in self._pending_order_ids:
            order = self._orders[order_id]
            age = self._minute - order.created_minute
            if age > self.config.cancel_after_minutes:
                order.status = "cancelled"
                self._step_reward -= 1.2
            else:
                kept_pending.append(order_id)
        self._pending_order_ids = kept_pending

        # idle/active accounting
        for courier in self._couriers.values():
            if courier.queue:
                courier.active_minutes += 1
            else:
                courier.idle_minutes += 1
                self._step_reward -= 0.003

        # spawn new orders for next decision point
        self._spawn_orders()
        self._minute += 1

    def _spawn_orders(self):
        rate = self._arrival_rate(self._minute)
        if self._rng.random() < self.config.burst_probability:
            rate *= 1.8
        arrivals = self._sample_poisson(rate)
        for _ in range(arrivals):
            order_id = f"O{self._next_order_id:06d}"
            self._next_order_id += 1
            restaurant_zone = self._sample_restaurant_zone()
            customer_zone = self._sample_customer_zone()
            prep = int(
                max(
                    3,
                    round(self._rng.gauss(self.config.mean_prep, self.config.std_prep)),
                )
            )
            created = self._minute
            order = Order(
                order_id=order_id,
                created_minute=created,
                restaurant_zone=restaurant_zone,
                customer_zone=customer_zone,
                prep_ready_minute=created + prep,
                deadline_minute=created + self.config.sla_minutes,
            )
            self._orders[order_id] = order
            self._pending_order_ids.append(order_id)

    def _arrival_rate(self, minute: int) -> float:
        # minute 0 corresponds to 10:00, emulate lunch and dinner peak
        hour = 10 + (minute // 60)
        multiplier = 1.0
        if 12 <= hour <= 14:
            multiplier *= self.config.lunch_peak_multiplier
        if 18 <= hour <= 21:
            multiplier *= self.config.dinner_peak_multiplier
        return self.config.base_arrival_rate * multiplier

    def _traffic_multiplier(self, minute: int) -> float:
        hour = 10 + (minute // 60)
        multiplier = 1.0
        if 12 <= hour <= 14 or 18 <= hour <= 21:
            multiplier *= self.config.traffic_peak_multiplier
        return multiplier

    def _estimate_completion(self, courier: Courier, order: Order) -> tuple[int, int]:
        if courier.queue:
            last_leg = max(courier.queue, key=lambda leg: leg.completion_minute)
            ready_minute = max(self._minute, last_leg.completion_minute)
            start_zone = last_leg.customer_zone
        else:
            ready_minute = max(self._minute, courier.reposition_finish_minute)
            start_zone = (
                courier.reposition_target_zone
                if courier.reposition_target_zone is not None
                else courier.zone
            )

        travel_to_rest = int(
            math.ceil(
                self._distance(start_zone, order.restaurant_zone)
                * self._traffic_multiplier(ready_minute)
            )
        )
        arrival_rest = ready_minute + travel_to_rest
        pickup_minute = max(arrival_rest, order.prep_ready_minute)
        travel_to_customer = int(
            math.ceil(
                self._distance(order.restaurant_zone, order.customer_zone)
                * self._traffic_multiplier(pickup_minute)
            )
        )
        completion = pickup_minute + travel_to_customer
        eta = completion - self._minute
        return eta, completion

    def _best_possible_eta(self, order: Order) -> int:
        best = None
        for courier in self._couriers.values():
            eta, _ = self._estimate_completion(courier, order)
            best = eta if best is None else min(best, eta)
        return best or 0

    def _courier_busy_for(self, courier: Courier) -> int:
        if not courier.queue:
            return 0
        return max(
            0, max(leg.completion_minute for leg in courier.queue) - self._minute
        )

    def _sample_restaurant_zone(self) -> int:
        hot_centers = [
            self._zone_id(int(self.config.width * 0.3), int(self.config.height * 0.4)),
            self._zone_id(int(self.config.width * 0.7), int(self.config.height * 0.6)),
        ]
        if self._rng.random() < 0.75:
            center = self._rng.choice(hot_centers)
            cx, cy = self._coord(center)
            x = min(self.config.width - 1, max(0, int(self._rng.gauss(cx, 2.0))))
            y = min(self.config.height - 1, max(0, int(self._rng.gauss(cy, 2.0))))
            return self._zone_id(x, y)
        return self._rng.randint(0, self.config.width * self.config.height - 1)

    def _sample_customer_zone(self) -> int:
        centers = [
            self._zone_id(int(self.config.width * 0.5), int(self.config.height * 0.5)),
            self._zone_id(int(self.config.width * 0.6), int(self.config.height * 0.3)),
        ]
        center = self._rng.choice(centers)
        cx, cy = self._coord(center)
        x = min(self.config.width - 1, max(0, int(self._rng.gauss(cx, 3.5))))
        y = min(self.config.height - 1, max(0, int(self._rng.gauss(cy, 3.5))))
        return self._zone_id(x, y)

    def _sample_poisson(self, rate: float) -> int:
        # Knuth for small rates, Gaussian approximation for larger rates
        if rate <= 8:
            l = math.exp(-rate)
            k = 0
            p = 1.0
            while p > l:
                k += 1
                p *= self._rng.random()
            return k - 1
        value = int(round(self._rng.gauss(rate, math.sqrt(rate))))
        return max(0, value)

    def _distance(self, zone_a: int, zone_b: int) -> int:
        ax, ay = self._coord(zone_a)
        bx, by = self._coord(zone_b)
        return abs(ax - bx) + abs(ay - by)

    def _coord(self, zone_id: int) -> tuple[int, int]:
        return zone_id % self.config.width, zone_id // self.config.width

    def _zone_id(self, x: int, y: int) -> int:
        return y * self.config.width + x
