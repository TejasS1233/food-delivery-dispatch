"""Gradio web interface for the Food Delivery Dispatch environment."""

from __future__ import annotations

import gradio as gr
import pandas as pd

from server.food_delivery_environment import SCENARIOS
from server.grader import run_policy_evaluation
from training.inference import list_registered_policies

HEURISTIC_POLICIES = ["nearest", "deadline", "hybrid"]


def get_all_policies() -> list[str]:
    policies = list(HEURISTIC_POLICIES)
    for rec in list_registered_policies():
        pid = rec.get("policy_id", "")
        if pid and pid not in policies:
            policies.append(pid)
    policies.append("auto_best")
    return policies


def run_single_evaluation(task_id: str, policy_id: str, episodes: int) -> tuple:
    """Run evaluation and return formatted results."""
    episodes = max(1, min(episodes, 20))
    m = run_policy_evaluation(
        task_id=task_id,
        policy_id=policy_id,
        episodes=episodes,
    )
    summary = (
        f"**Task:** {task_id}\n\n"
        f"**Policy:** {policy_id}\n\n"
        f"**Score:** {m.score:.4f} / 1.0\n\n"
        f"| Metric | Value |\n"
        f"|---|---|\n"
        f"| On-time Rate | {m.on_time_rate:.1%} |\n"
        f"| Cancellation Rate | {m.cancellation_rate:.1%} |\n"
        f"| Rejection Rate | {m.rejection_rate:.1%} |\n"
        f"| Avg Delivery Time | {m.avg_delivery_minutes:.1f} min |\n"
        f"| Courier Utilization | {m.courier_utilization:.1%} |\n"
        f"| Fairness Score | {m.fairness_score:.3f} |\n"
    )
    return summary, m.score


def run_full_baseline(episodes: int) -> tuple:
    """Run baseline across all tasks and policies."""
    episodes = max(1, min(episodes, 20))
    rows = []
    for task in SCENARIOS:
        for policy in get_all_policies():
            m = run_policy_evaluation(
                task_id=task,
                policy_id=policy,
                episodes=episodes,
            )
            rows.append(
                {
                    "Task": task,
                    "Policy": policy,
                    "Score": round(m.score, 4),
                    "On-Time %": round(m.on_time_rate * 100, 1),
                    "Cancel %": round(m.cancellation_rate * 100, 1),
                    "Reject %": round(m.rejection_rate * 100, 1),
                    "Avg Delivery (min)": round(m.avg_delivery_minutes, 1),
                    "Utilization %": round(m.courier_utilization * 100, 1),
                }
            )
    df = pd.DataFrame(rows)
    return df


def compare_policies(task_id: str, episodes: int) -> pd.DataFrame:
    """Compare all policies on a single task."""
    episodes = max(1, min(episodes, 20))
    rows = []
    for policy in get_all_policies():
        m = run_policy_evaluation(
            task_id=task_id,
            policy_id=policy,
            episodes=episodes,
        )
        rows.append(
            {
                "Policy": policy,
                "Score": round(m.score, 4),
                "On-Time %": round(m.on_time_rate * 100, 1),
                "Cancel %": round(m.cancellation_rate * 100, 1),
                "Reject %": round(m.rejection_rate * 100, 1),
                "Avg Delivery (min)": round(m.avg_delivery_minutes, 1),
                "Utilization %": round(m.courier_utilization * 100, 1),
                "Fairness": round(m.fairness_score, 3),
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("Score", ascending=False)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Food Delivery Dispatch", theme=gr.themes.Soft()) as ui:
        gr.Markdown("# Food Delivery Dispatch Environment")
        gr.Markdown(
            "Interactive evaluation interface for the food delivery dispatch "
            "simulation. Run policies, compare scores, and explore the environment."
        )

        with gr.Tabs():
            with gr.Tab("Single Evaluation"):
                with gr.Row():
                    with gr.Column():
                        eval_task = gr.Dropdown(
                            choices=list(SCENARIOS.keys()),
                            value="medium",
                            label="Task",
                        )
                        eval_policy = gr.Dropdown(
                            choices=get_all_policies(),
                            value="hybrid",
                            label="Policy",
                        )
                        eval_episodes = gr.Slider(
                            minimum=1, maximum=20, value=3, step=1, label="Episodes"
                        )
                        eval_btn = gr.Button("Run Evaluation", variant="primary")
                    with gr.Column():
                        eval_output = gr.Markdown(label="Results")
                        eval_score = gr.Number(label="Score", interactive=False)

                eval_btn.click(
                    fn=run_single_evaluation,
                    inputs=[eval_task, eval_policy, eval_episodes],
                    outputs=[eval_output, eval_score],
                )

            with gr.Tab("Policy Comparison"):
                with gr.Row():
                    with gr.Column(scale=1):
                        comp_task = gr.Dropdown(
                            choices=list(SCENARIOS.keys()),
                            value="medium",
                            label="Task",
                        )
                        comp_episodes = gr.Slider(
                            minimum=1, maximum=20, value=3, step=1, label="Episodes"
                        )
                        comp_btn = gr.Button("Compare All Policies", variant="primary")
                    with gr.Column(scale=3):
                        comp_output = gr.Dataframe(
                            label="Policy Comparison", interactive=False
                        )

                comp_btn.click(
                    fn=compare_policies,
                    inputs=[comp_task, comp_episodes],
                    outputs=[comp_output],
                )

            with gr.Tab("Full Baseline"):
                with gr.Row():
                    with gr.Column(scale=1):
                        bl_episodes = gr.Slider(
                            minimum=1, maximum=20, value=3, step=1, label="Episodes"
                        )
                        bl_btn = gr.Button("Run Full Baseline", variant="primary")
                    with gr.Column(scale=3):
                        bl_output = gr.Dataframe(
                            label="Baseline Results", interactive=False
                        )

                bl_btn.click(
                    fn=run_full_baseline,
                    inputs=[bl_episodes],
                    outputs=[bl_output],
                )

        gr.Markdown("---")
        gr.Markdown(
            "**Policies:** nearest (greedy), deadline (SLA-aware), hybrid (adaptive), "
            "ddqn_per_v1 (trained DDQN), ppo_masked_v1 (trained PPO), auto_best (routing), "
            "llm (LLM agent via OpenAI API)"
        )

    return ui


def mount(app):
    """Mount the Gradio app onto the FastAPI app at /web."""
    gradio_app = build_ui()
    gr.mount_gradio_app(app, gradio_app, path="/web")
