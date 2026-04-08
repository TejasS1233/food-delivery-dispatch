#!/usr/bin/env python

from __future__ import annotations

import argparse
from collections import deque
import random

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "PyTorch is required for training.train_ddqn_per. Install dependencies with: uv sync"
    ) from exc

from decision import action_mask, choose_meta_action, META_ACTIONS
from server.food_delivery_environment import FoodDeliveryDispatchEnvironment
from training.common import MODELS_DIR, obs_to_vector, register_policy


class QNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PrioritizedReplay:
    def __init__(self, capacity: int = 20000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: deque = deque(maxlen=capacity)
        self.priorities: deque = deque(maxlen=capacity)

    def push(self, transition, priority: float = 1.0):
        self.buffer.append(transition)
        self.priorities.append(float(priority))

    def sample(self, batch_size: int, beta: float = 0.4):
        priorities = np.array(self.priorities, dtype=np.float32)
        scaled = np.power(priorities + 1e-6, self.alpha)
        probs = scaled / scaled.sum()
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        weights = np.power(len(self.buffer) * probs[indices], -beta)
        weights = weights / weights.max()
        return indices, samples, weights

    def update_priorities(self, indices, td_errors):
        for i, e in zip(indices, td_errors):
            self.priorities[i] = float(abs(e) + 1e-3)

    def __len__(self):
        return len(self.buffer)


def select_action(
    qnet: QNet, state_vec: list[float], mask: list[int], epsilon: float
) -> int:
    if random.random() < epsilon:
        valid = [i for i, m in enumerate(mask) if m == 1]
        return random.choice(valid) if valid else 0

    with torch.no_grad():
        x = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
        q_values = qnet(x).squeeze(0).cpu().numpy()
    masked = np.where(np.array(mask) == 1, q_values, -1e9)
    return int(masked.argmax())


def train(args):
    tasks = ["easy", "medium", "hard"]
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    probe_env = FoodDeliveryDispatchEnvironment(task="easy")
    probe_obs = probe_env.reset(task="easy")
    obs_dim = len(obs_to_vector(probe_obs))
    action_dim = len(META_ACTIONS)

    qnet = QNet(obs_dim, action_dim)
    target = QNet(obs_dim, action_dim)
    target.load_state_dict(qnet.state_dict())

    optimizer = optim.Adam(qnet.parameters(), lr=args.lr)
    replay = PrioritizedReplay(capacity=args.replay_size)

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    beta = 0.4

    for episode in range(args.episodes):
        task_id = tasks[episode % len(tasks)]
        env = FoodDeliveryDispatchEnvironment(task=task_id)
        obs = env.reset(task=task_id)

        ep_reward = 0.0
        while not obs.done:
            s_vec = obs_to_vector(obs)
            mask = action_mask(obs)
            action_id = select_action(qnet, s_vec, mask, epsilon)

            env_action = choose_meta_action(action_id, obs).action
            next_obs = env.step(env_action)
            r = next_obs.reward
            done = next_obs.done
            next_vec = obs_to_vector(next_obs)
            next_mask = action_mask(next_obs)
            replay.push(
                (s_vec, action_id, r, next_vec, done, mask, next_mask), priority=1.0
            )

            obs = next_obs
            ep_reward += r

            if len(replay) >= args.batch_size:
                beta = min(1.0, beta + 1e-4)
                idx, batch, weights = replay.sample(args.batch_size, beta=beta)
                states = torch.tensor([b[0] for b in batch], dtype=torch.float32)
                actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
                rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
                next_states = torch.tensor([b[3] for b in batch], dtype=torch.float32)
                dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)
                next_masks = torch.tensor([b[6] for b in batch], dtype=torch.float32)
                w = torch.tensor(weights, dtype=torch.float32)

                q = qnet(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q_online = qnet(next_states)
                    next_q_online = next_q_online.masked_fill(next_masks == 0, -1e9)
                    next_a = next_q_online.argmax(dim=1)
                    next_q_target = (
                        target(next_states).gather(1, next_a.unsqueeze(1)).squeeze(1)
                    )
                    y = rewards + args.gamma * (1 - dones) * next_q_target

                td_error = (q - y).detach().cpu().numpy()
                loss = (w * (q - y).pow(2)).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(qnet.parameters(), 1.0)
                optimizer.step()

                replay.update_priorities(idx, td_error)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if (episode + 1) % args.target_update == 0:
            target.load_state_dict(qnet.state_dict())

        if (episode + 1) % args.log_interval == 0:
            print(
                f"episode={episode + 1}/{args.episodes} task={task_id} reward={ep_reward:.2f} epsilon={epsilon:.3f}"
            )

    policy_id = args.policy_id or f"ddqn_per_{args.episodes}ep"
    checkpoint = MODELS_DIR / f"{policy_id}.pt"
    torch.save(
        {
            "algo": "ddqn_per",
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "state_dict": qnet.state_dict(),
            "meta_actions": META_ACTIONS,
        },
        checkpoint,
    )

    register_policy(
        policy_id=policy_id,
        algo="ddqn_per",
        checkpoint_path=checkpoint,
        task_mix=tasks,
        notes="Meta-action DDQN with prioritized replay over dispatch heuristics/actions",
    )
    print(f"saved policy: {policy_id} -> {checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=20000)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-update", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy-id", type=str, default="")
    train(parser.parse_args())
