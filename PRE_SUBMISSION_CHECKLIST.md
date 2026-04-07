# Pre-Submission Checklist (Realistic Status)

This file tracks the organizer's hard requirements and current project status.

## 1) HF Space deploys
- [ ] Space URL returns `200` and responds to `POST /reset`
- Status: Not yet validated against a live HF Space URL.
- How to verify:
  - Deploy Space
  - Run: `./validate-submission.sh https://<your-space>.hf.space .`

## 2) OpenEnv spec compliance
- [x] `openenv.yaml` present and points to app entry (`server.app:app`)
- [x] Typed models implemented (`FoodDeliveryAction`, `FoodDeliveryObservation`)
- [x] Environment implements `reset()/step()/state()`
- [x] `openenv validate` passes locally

## 3) Dockerfile builds
- [x] Root `Dockerfile` added
- [x] `server/Dockerfile` present (legacy alternate)
- [ ] Docker build confirmed on this machine
- Status note: local Docker daemon was not running during check.
- How to verify once daemon is running:
  - `docker build .`

## 4) Baseline reproduces (inference.py)
- [x] `inference.py` exists at project root
- [x] Uses OpenAI client for all LLM calls
- [x] Supports required env vars: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- [x] Structured stdout format `[START]`, `[STEP]`, `[END]`
- [x] Local dry-run completed without error and emitted score in `[0,1]`
- [x] Fallback path works without LLM key (uses deterministic dispatch fallback)

## 5) 3+ tasks with graders
- [x] Tasks: `easy`, `medium`, `hard`
- [x] `/grader` implemented
- [x] Scores in `[0,1]` (verified locally)

## 6) Mandatory additional instructions
- [x] Inference script name is exactly `inference.py`
- [x] OpenAI client usage for LLM calls
- [x] STDOUT fields follow required ordering and formatting
- [ ] Confirm deployment config includes all three vars (`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`) in Space settings

## 7) Infra restrictions
- [x] Inference runtime path is lightweight (`MAX_STEPS` bounded, simple loop)
- [x] Designed to run under modest resources (2 vCPU / 8GB expected)
- [ ] Validate actual wall-clock on deployed Space (target <20 minutes)

## 8) Test and quality gates
- [x] Pytest suite passes (`6 passed`)
- [x] `openenv validate` passes
- [ ] Docker build pass with active local daemon

## Final remaining blockers before submission
1. Deploy HF Space and verify `/reset` returns `200`.
2. Run `validate-submission.sh` against live Space URL and capture pass output.
3. Start Docker daemon and verify local `docker build .` passes.
4. Ensure Space secrets/variables include `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`.
