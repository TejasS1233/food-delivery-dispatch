from server.food_delivery_environment import FoodDeliveryDispatchEnvironment
from server.grader import run_policy_evaluation
from models import FoodDeliveryAction


def test_reset_is_clean_and_step_advances():
    env = FoodDeliveryDispatchEnvironment(task="easy")
    obs = env.reset(task="easy")
    assert obs.minute == 0
    assert obs.task_id == "easy"
    assert not obs.done

    obs2 = env.step(FoodDeliveryAction(action_type="wait"))
    assert obs2.minute == 1


def test_grader_scores_in_range():
    for task in ["easy", "medium", "hard"]:
        m = run_policy_evaluation(task_id=task, policy_id="hybrid", episodes=1)
        assert 0.0 <= m.score <= 1.0


def test_grader_is_deterministic_with_fixed_seed():
    a = run_policy_evaluation(task_id="hard", policy_id="hybrid", episodes=1).score
    b = run_policy_evaluation(task_id="hard", policy_id="hybrid", episodes=1).score
    assert a == b
