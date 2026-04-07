from typing import Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


ActionType = Literal["assign", "reject", "reposition", "wait"]


class PendingOrderView(BaseModel):
    order_id: str
    restaurant_zone: int
    customer_zone: int
    age_minutes: int
    prep_remaining_minutes: int
    sla_remaining_minutes: int
    estimated_best_eta: int


class CourierView(BaseModel):
    courier_id: str
    zone: int
    busy_for_minutes: int
    queue_length: int
    reposition_remaining_minutes: int


class FoodDeliveryAction(Action):
    action_type: ActionType = Field(..., description="Dispatch action to apply")
    order_id: str | None = Field(default=None, description="Order to assign or reject")
    courier_id: str | None = Field(
        default=None, description="Courier to assign or reposition"
    )
    target_zone: int | None = Field(
        default=None, description="Target zone for repositioning"
    )


class FoodDeliveryObservation(Observation):
    task_id: str = Field(..., description="Task difficulty id")
    minute: int = Field(..., description="Current minute of episode")
    horizon: int = Field(..., description="Episode horizon in minutes")
    pending_orders: list[PendingOrderView] = Field(default_factory=list)
    couriers: list[CourierView] = Field(default_factory=list)
    total_orders_created: int = 0
    total_delivered: int = 0
    total_on_time: int = 0
    total_late: int = 0
    total_rejected: int = 0
    total_cancelled: int = 0
    average_delivery_minutes: float = 0.0
    cumulative_reward: float = 0.0
    done: bool = False
    reward: float = 0.0
    metadata: dict = Field(default_factory=dict)
