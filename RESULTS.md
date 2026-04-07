# Benchmark Results (5 Episodes per Task)

Evaluation command used:

```bash
uv run python -c "import json; from server.grader import run_policy_evaluation; tasks=['easy','medium','hard']; policies=['nearest','deadline','hybrid','ddqn_per_v1','ppo_masked_v1']; out=[]; \
for t in tasks:\
  for p in policies:\
    m=run_policy_evaluation(t,p,episodes=5); out.append((t,p,m.score,m.on_time_rate,m.cancellation_rate,m.rejection_rate,m.avg_delivery_minutes)); \
print(json.dumps(out, indent=2))"
```

## Summary

- `ddqn_per_v1` is strongest on `easy` and `medium`.
- `hybrid` heuristic is strongest on `hard` in current checkpoints.
- `ppo_masked_v1` underperforms and should not be primary.

## Detailed Table

| Task | Policy | Score | On-Time | Cancel | Reject | Avg Delivery (min) |
|---|---|---:|---:|---:|---:|---:|
| easy | nearest | 0.8575 | 0.9885 | 0.0000 | 0.0000 | 18.24 |
| easy | deadline | 0.8573 | 0.9885 | 0.0000 | 0.0000 | 18.29 |
| easy | hybrid | 0.8634 | 1.0000 | 0.0000 | 0.0099 | 17.97 |
| easy | ddqn_per_v1 | **0.9054** | 0.9881 | 0.0000 | 0.0495 | 18.52 |
| easy | ppo_masked_v1 | 0.8573 | 0.9885 | 0.0000 | 0.0000 | 18.29 |
| medium | nearest | 0.7655 | 0.8561 | 0.0000 | 0.0000 | 24.73 |
| medium | deadline | 0.7615 | 0.8615 | 0.0281 | 0.0000 | 24.58 |
| medium | hybrid | 0.8065 | 0.9286 | 0.0112 | 0.0337 | 23.11 |
| medium | ddqn_per_v1 | **0.8644** | 0.9328 | 0.0000 | 0.1180 | 22.19 |
| medium | ppo_masked_v1 | 0.7615 | 0.8615 | 0.0281 | 0.0000 | 24.58 |
| hard | nearest | 0.4206 | 0.2754 | 0.2287 | 0.0000 | 51.65 |
| hard | deadline | 0.3825 | 0.2593 | 0.2867 | 0.0000 | 47.76 |
| hard | hybrid | **0.6980** | 0.7105 | 0.0717 | 0.1911 | 25.78 |
| hard | ddqn_per_v1 | 0.6390 | 0.5714 | 0.0239 | 0.2355 | 30.05 |
| hard | ppo_masked_v1 | 0.3825 | 0.2593 | 0.2867 | 0.0000 | 47.76 |

## Recommended Submission Policy

- `auto_best` composite policy:
  - `easy`/`medium` -> `ddqn_per_v1`
  - `hard` -> `hybrid`
