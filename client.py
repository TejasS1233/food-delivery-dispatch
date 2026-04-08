"""Food Delivery Dispatch Environment Client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import FoodDeliveryAction, FoodDeliveryObservation
except ImportError:  # pragma: no cover
    from models import FoodDeliveryAction, FoodDeliveryObservation


class FoodDeliveryEnv(EnvClient[FoodDeliveryAction, FoodDeliveryObservation, State]):
    """
    Client for the Food Delivery Dispatch Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with FoodDeliveryEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset(task="easy")
        ...     print(result.observation.pending_orders)
        ...
        ...     result = client.step(FoodDeliveryAction(
        ...         action_type="assign",
        ...         order_id="O000001",
        ...         courier_id="C01",
        ...     ))
        ...     print(result.observation.total_delivered)

    Example with Docker:
        >>> client = FoodDeliveryEnv.from_docker_image("food-delivery-dispatch:latest")
        >>> try:
        ...     result = client.reset(task="medium")
        ...     result = client.step(FoodDeliveryAction(action_type="wait"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: FoodDeliveryAction) -> Dict[str, Any]:
        """
        Convert FoodDeliveryAction to JSON payload for step message.

        Args:
            action: FoodDeliveryAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload: Dict[str, Any] = {"action_type": action.action_type}
        if action.order_id is not None:
            payload["order_id"] = action.order_id
        if action.courier_id is not None:
            payload["courier_id"] = action.courier_id
        if action.target_zone is not None:
            payload["target_zone"] = action.target_zone
        return payload

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[FoodDeliveryObservation]:
        """
        Parse server response into StepResult[FoodDeliveryObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with FoodDeliveryObservation
        """
        obs_data = payload.get("observation", {})
        observation = FoodDeliveryObservation(
            task_id=obs_data.get("task_id", ""),
            minute=obs_data.get("minute", 0),
            horizon=obs_data.get("horizon", 0),
            pending_orders=obs_data.get("pending_orders", []),
            couriers=obs_data.get("couriers", []),
            total_orders_created=obs_data.get("total_orders_created", 0),
            total_delivered=obs_data.get("total_delivered", 0),
            total_on_time=obs_data.get("total_on_time", 0),
            total_late=obs_data.get("total_late", 0),
            total_rejected=obs_data.get("total_rejected", 0),
            total_cancelled=obs_data.get("total_cancelled", 0),
            average_delivery_minutes=obs_data.get("average_delivery_minutes", 0.0),
            cumulative_reward=obs_data.get("cumulative_reward", 0.0),
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward", 0.0)),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
