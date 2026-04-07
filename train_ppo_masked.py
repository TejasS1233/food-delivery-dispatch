#!/usr/bin/env python

from __future__ import annotations

import argparse
import random

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "PyTorch is required for train_ppo_masked.py. Install with: uv sync --extra train"
    ) from exc

from decision import action_mask, choose_meta_action, META_ACTIONS
from server.food_delivery_environment import FoodDeliveryDispatchEnvironment
from training.common import MODELS_DIR, obs_to_vector, register_policy


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        h = self.base(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(mask <= 0, -1e9)


def train(args):
    tasks = ["easy", "medium", "hard"]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    probe_env = FoodDeliveryDispatchEnvironment(task="easy")
    probe_obs = probe_env.reset(task="easy")
    obs_dim = len(obs_to_vector(probe_obs))
    action_dim = len(META_ACTIONS)

    model = ActorCritic(obs_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for update in range(args.updates):
        storage = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "dones": [],
            "values": [],
            "masks": [],
        }

        episode_rewards = []
        for ep in range(args.episodes_per_update):
            task_id = tasks[(update * args.episodes_per_update + ep) % len(tasks)]
            env = FoodDeliveryDispatchEnvironment(task=task_id)
            obs = env.reset(task=task_id)
            ep_reward = 0.0

            while not obs.done:
                s_vec = torch.tensor(obs_to_vector(obs), dtype=torch.float32).unsqueeze(
                    0
                )
                m_vec = torch.tensor(action_mask(obs), dtype=torch.float32).unsqueeze(0)
                logits, value = model(s_vec)
                logits = masked_logits(logits, m_vec)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                env_action = choose_meta_action(int(action.item()), obs).action
                next_obs = env.step(env_action)

                storage["states"].append(s_vec.squeeze(0))
                storage["actions"].append(
                    action.squeeze(0) if action.ndim > 0 else action
                )
                storage["log_probs"].append(
                    log_prob.squeeze(0) if log_prob.ndim > 0 else log_prob
                )
                storage["rewards"].append(float(next_obs.reward))
                storage["dones"].append(float(next_obs.done))
                storage["values"].append(value.squeeze(0) if value.ndim > 0 else value)
                storage["masks"].append(m_vec.squeeze(0))

                obs = next_obs
                ep_reward += next_obs.reward

            episode_rewards.append(ep_reward)

        with torch.no_grad():
            returns = []
            advantages = []
            next_value = 0.0
            gae = 0.0

            values = [v.item() for v in storage["values"]]
            rewards = storage["rewards"]
            dones = storage["dones"]

            for t in reversed(range(len(rewards))):
                delta = (
                    rewards[t] + args.gamma * next_value * (1.0 - dones[t]) - values[t]
                )
                gae = delta + args.gamma * args.lam * (1.0 - dones[t]) * gae
                adv = gae
                ret = adv + values[t]

                advantages.append(adv)
                returns.append(ret)
                next_value = values[t]

            advantages.reverse()
            returns.reverse()

        states = torch.stack(storage["states"])
        actions = torch.tensor(storage["actions"], dtype=torch.long)
        old_log_probs = torch.stack(storage["log_probs"]).detach()
        returns_t = torch.tensor(returns, dtype=torch.float32)
        adv_t = torch.tensor(advantages, dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        masks = torch.stack(storage["masks"])

        n = len(states)
        idxs = np.arange(n)
        for _ in range(args.epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, args.batch_size):
                end = start + args.batch_size
                mb = idxs[start:end]

                logits, values = model(states[mb])
                logits = masked_logits(logits, masks[mb])
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions[mb])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv_t[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (returns_t[mb] - values).pow(2).mean()

                loss = (
                    policy_loss
                    + args.value_coef * value_loss
                    - args.entropy_coef * entropy
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        if (update + 1) % args.log_interval == 0:
            print(
                f"update={update + 1}/{args.updates} avg_episode_reward={np.mean(episode_rewards):.2f}"
            )

    policy_id = args.policy_id or f"ppo_masked_{args.updates}u"
    checkpoint = MODELS_DIR / f"{policy_id}.pt"
    torch.save(
        {
            "algo": "ppo_masked",
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "state_dict": model.state_dict(),
            "meta_actions": META_ACTIONS,
        },
        checkpoint,
    )

    register_policy(
        policy_id=policy_id,
        algo="ppo_masked",
        checkpoint_path=checkpoint,
        task_mix=tasks,
        notes="Masked PPO over high-level dispatch action templates",
    )
    print(f"saved policy: {policy_id} -> {checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument("--episodes-per-update", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--policy-id", type=str, default="")
    train(parser.parse_args())
