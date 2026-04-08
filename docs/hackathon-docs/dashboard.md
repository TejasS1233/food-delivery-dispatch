# Round 1 — Problem Statement

## The Task
Build a complete, real-world OpenEnv environment that an AI agent can learn from through the standard `step()` / `reset()` / `state()` API.

## Key Requirements at a Glance
* **Must simulate a real-world task** (not games or toys).
* **Implement full OpenEnv spec:** typed models, `step()` / `reset()` / `state()`, `openenv.yaml`.
* **Minimum 3 tasks with agent graders** (easy → medium → hard, scores 0.0–1.0).
* **Meaningful reward function** with partial progress signals.
* **Baseline inference script** with reproducible scores.
* **Deploy to Hugging Face Spaces** + working Dockerfile.
* **README** with environment description, action/observation spaces, and setup instructions.

---

## Functional Requirements

* **Real-world task simulation:** The environment must simulate a task humans actually do. Not games, not toys. Examples: email triage, code review, data cleaning, scheduling, customer support, content moderation.
* **OpenEnv spec compliance:** Implement the full OpenEnv interface: typed Observation, Action, and Reward Pydantic models. 
    * `step(action)` → returns observation, reward, done, info. 
    * `reset()` → returns initial observation. 
    * `state()` → returns current state. 
    * `openenv.yaml` with metadata. Tested via `openenv validate`.
* **Minimum 3 tasks with agent graders:** Each task defines a concrete objective an agent must accomplish, with a programmatic grader that scores performance (0.0–1.0). Tasks should range: easy → medium → hard. Graders must have clear, deterministic success/failure criteria.
* **Meaningful reward function:** Provides signal over the full trajectory (not just binary end-of-episode). Rewards partial progress toward task completion. Penalizes clearly undesirable behavior (e.g., infinite loops, destructive actions).
* **Baseline inference script:** Uses the OpenAI API client to run a model against the environment. Reads API credentials from environment variables (`OPENAI_API_KEY`). Produces a reproducible baseline score on all 3 tasks.

## Non-Functional Requirements

* **Deploys to a Hugging Face Space:** Environment must run as a containerized HF Space tagged with `openenv`.
* **Containerized execution:** Must include a working Dockerfile. The environment should start cleanly with `docker build` + `docker run`.
* **Documentation:** README must include: environment description and motivation, action and observation space definitions, task descriptions with expected difficulty, setup and usage instructions, baseline scores.

---

## Evaluation Criteria

| Parameter | Weight | Description |
| :--- | :--- | :--- |
| **Real-world utility** | 30% | Does the environment model a genuine task? Would someone actually use this to train or evaluate agents? |
| **Task & grader quality** | 25% | Are tasks well-defined with clear objectives? Do graders accurately and fairly measure success? Meaningful difficulty progression? |
| **Environment design** | 20% | Clean state management, sensible action/observation spaces, good reward shaping, proper episode boundaries. |
| **Code quality & spec compliance** | 15% | Follows OpenEnv spec, clean project structure, typed models, documented, tested, Dockerfile works. |
| **Creativity & novelty** | 10% | Novel problem domain, interesting mechanics, clever reward design, original approach. |

### Scoring Breakdown

**Real-world utility (30%)**
* **0–5:** Toy/artificial problem with no practical application.
* **6–15:** Valid domain but shallow modeling of the real task.
* **16–25:** Good domain modeling, would be useful for agent evaluation.
* **26–30:** Excellent — fills a real gap, immediate value for the RL/agent community.

**Task & grader quality (25%)**
* 3+ tasks with difficulty range?
* Graders produce scores between 0.0–1.0?
* Graders deterministic and reproducible?
* Hard task genuinely challenges frontier models?

**Environment design (20%)**
* `reset()` produces clean state?
* Action/observation types well-designed and documented?
* Reward function provides useful varying signal (not just sparse)?
* Episode boundaries sensible?

**Code quality & spec compliance (15%)**
* `openenv validate` passes?
* `docker build` && `docker run` works?
* HF Space deploys and responds?
* Baseline script runs and reproduces scores?

**Creativity & novelty (10%)**
* Domain we haven’t seen in OpenEnv before?
* Reward design has interesting properties?
* Clever mechanics that make the environment engaging?

---

## How Judging Works

1.  **Phase 1: Automated Validation:** Pass/fail gate — HF Space deploys, OpenEnv spec compliance, Dockerfile builds, baseline reproduces, 3+ tasks with graders.
2.  **Phase 2: Agentic Evaluation:** Scored — baseline agent re-run, standard Open LLM agent (e.g., Nemotron 3 Super) run against all environments, score variance check.
3.  **Phase 3: Human Review:** Top submissions reviewed by Meta and Hugging Face engineers for real-world utility, creativity, and exploit checks.

### Disqualification Criteria
* Environment does not deploy or respond.
* Plagiarized or trivially modified existing environments.
* Graders that always return the same score.
* No baseline inference script.

---

## Pre-Submission Checklist 
*(All must pass or you're disqualified)*

* **HF Space deploys:** Automated ping to the Space URL — must return 200 and respond to `reset()`.
* **OpenEnv spec compliance:** Validate `openenv.yaml`, typed models, `step()` / `reset()` / `state()` endpoints.
* **Dockerfile builds:** Automated `docker build` on the submitted repo.
* **Baseline reproduces:** Run the submitted inference script — must complete without error and produce scores.
* **3+ tasks with graders:** Enumerate tasks, run each grader, verify scores in 0.0–1.0 range.

### Additional Endpoints to Expose
* `/baseline`: Trigger inference script and returns baseline score for all 3 tasks.
* `/grader`: Returns grader score after an episode is completed.
* `/tasks`: Returns list of tasks and the action schema (fields required for an action in a step).

### Validator
Run the pre-submission validation script before submitting.

---

# Round 1 Guide

## What to Expect & How to Submit
When Round 1 opens, you'll choose 1 of 4–5 problem statements and build an OpenEnv environment around it.

### Example Problem Statement Workflow
*"Build a mini-game RL environment with clearly defined tasks, automated graders, and reward logic using the OpenEnv framework."*

1.  Create a mini-game an AI agent can play.
2.  Define tasks with increasing difficulty.
3.  Write graders that verify task completion.
4.  Define reward logic for scoring.
5.  Package using OpenEnv for automated evaluation.

### Evaluation Criteria (Summary)
* **Runtime correctness:** Runs without errors.
* **Interface compliance:** Follows OpenEnv standard.
* **Task design:** Clear, realistic, testable.
* **Grading logic:** Reward system makes sense.