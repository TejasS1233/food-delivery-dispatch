from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from decision import META_ACTIONS, action_mask
from models import FoodDeliveryObservation
from training.common import load_registry, obs_to_vector


@dataclass
class LoadedPolicy:
    policy_id: str
    algo: str
    checkpoint_path: str


def list_registered_policies() -> list[dict]:
    return load_registry().get("policies", [])


def get_policy_record(policy_id: str) -> dict | None:
    for rec in list_registered_policies():
        if rec.get("policy_id") == policy_id:
            return rec
    return None


def predict_meta_action(policy_id: str, obs: FoodDeliveryObservation) -> int:
    rec = get_policy_record(policy_id)
    if rec is None:
        raise ValueError(f"Policy not found in registry: {policy_id}")

    algo = rec.get("algo")
    ckpt_path = Path(rec.get("checkpoint_path", ""))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for trained policy inference. Install torch in this environment."
        ) from exc

    payload = torch.load(ckpt_path, map_location="cpu")
    obs_vec = torch.tensor(obs_to_vector(obs), dtype=torch.float32).unsqueeze(0)
    mask = np.array(action_mask(obs), dtype=np.int32)

    action_dim = len(META_ACTIONS)
    obs_dim = int(payload.get("obs_dim", obs_vec.shape[1]))
    if obs_dim != obs_vec.shape[1]:
        raise ValueError(
            f"Observation dimension mismatch: checkpoint={obs_dim}, runtime={obs_vec.shape[1]}"
        )

    if algo == "ddqn_per":

        class QNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(obs_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, action_dim),
                )

            def forward(self, x):
                return self.net(x)

        model = QNet()
        model.load_state_dict(payload["state_dict"])
        model.eval()
        with torch.no_grad():
            q_vals = model(obs_vec).squeeze(0).numpy()
        q_vals = np.where(mask == 1, q_vals, -1e9)
        return int(np.argmax(q_vals))

    if algo == "ppo_masked":

        class AC(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Sequential(
                    nn.Linear(obs_dim, 128),
                    nn.Tanh(),
                    nn.Linear(128, 128),
                    nn.Tanh(),
                )
                self.policy_head = nn.Linear(128, action_dim)
                self.value_head = nn.Linear(128, 1)

            def forward(self, x):
                h = self.base(x)
                return self.policy_head(h), self.value_head(h)

        model = AC()
        model.load_state_dict(payload["state_dict"])
        model.eval()
        with torch.no_grad():
            logits, _ = model(obs_vec)
            logits = logits.squeeze(0).numpy()
        logits = np.where(mask == 1, logits, -1e9)
        return int(np.argmax(logits))

    raise ValueError(f"Unsupported policy algo: {algo}")
