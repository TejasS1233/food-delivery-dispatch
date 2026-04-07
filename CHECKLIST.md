# Food Delivery Env 2.0 Checklist

## 0) Scope Guardrails
- [x] Build in a separate folder (`food_delivery_env_v2/`)
- [x] Keep old `my_env/` untouched
- [x] No mocked endpoints that bypass environment dynamics

## 1) Real-World Utility (30%)
- [x] Model minute-level operations: order arrivals, prep delays, courier movement/availability
- [x] Include hard trade-offs: on-time delivery, cancellations, courier utilization, idle movement cost
- [x] Support demand peaks (lunch/dinner) and traffic variation over time
- [x] Include practical controls: assign, reject, reposition, wait
- [x] Validate with deterministic scenario evaluations and KPI tables

## 2) Task & Grader Quality (25%)
- [x] Provide 3+ tasks with clear progression (`easy`, `medium`, `hard`)
- [x] Ensure graders return score in `[0.0, 1.0]`
- [x] Use deterministic fixed seeds for reproducible grader outputs
- [x] Hard task includes stress regime (burst demand + tighter SLA + lower courier slack)
- [x] Publish grader formula and metric normalization in README

## 3) Environment Design (20%)
- [x] `reset()` returns clean episode state
- [x] `step()` mutates state once per minute and records transitions
- [x] Typed `Action` and `Observation` models with documented fields
- [x] Dense, multi-objective reward (not sparse binary)
- [x] Sensible episode boundaries (`horizon` reached or terminal collapse condition)
- [x] Action validity checks + invalid action penalties

## 4) Code Quality & Spec Compliance (15%)
- [x] `openenv.yaml` added
- [x] FastAPI app exposes required endpoints (`/reset`, `/step`, `/tasks`, `/baseline`, `/grader`)
- [ ] Dockerfile builds and runs server
- [x] Baseline script added (`run_baseline.py`)
- [x] Project structure is clean and typed
- [x] README includes setup, run, evaluation, and expected outputs

## 5) Creativity & Novelty (10%)
- [x] Add non-trivial mechanic: strategic order rejection + proactive repositioning
- [x] Add fairness-aware metric in grader (utilization balance)
- [x] Include a stress test scenario not seen in default training

## 6) Training Variants
- [x] Heuristic baselines (`nearest`, `deadline`, `hybrid`)
- [x] `ddqn_per` training pipeline
- [x] `ppo_masked` training pipeline
- [x] Standardized evaluation script across all policies
- [x] Keep training offline via scripts (no long-running `/train` API)

## 7) Done Criteria
- [x] End-to-end local run: server up, episode run, grader score emitted
- [x] Deterministic re-run gives identical task scores for same policy and seeds
- [x] Baseline report generated for all tasks
