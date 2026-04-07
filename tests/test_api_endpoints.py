from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


def test_tasks_endpoint():
    r = client.get("/tasks")
    assert r.status_code == 200
    payload = r.json()
    assert len(payload["tasks"]) >= 3


def test_grader_endpoint_score_range():
    r = client.post(
        "/grader", json={"task_id": "medium", "policy_id": "hybrid", "episodes": 1}
    )
    assert r.status_code == 200
    score = r.json()["score"]
    assert 0.0 <= score <= 1.0


def test_evaluate_composite_policy():
    r = client.post(
        "/evaluate", json={"task_id": "hard", "policy_id": "auto_best", "episodes": 1}
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["policy_id"] == "hybrid"
    assert 0.0 <= payload["score"] <= 1.0
