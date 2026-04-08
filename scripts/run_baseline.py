#!/usr/bin/env python

import argparse
import json
import requests


def main():
    parser = argparse.ArgumentParser(description="Run food delivery baseline report")
    parser.add_argument(
        "--url", default="http://localhost:8000", help="Server base URL"
    )
    parser.add_argument(
        "--episodes", type=int, default=3, help="Episodes per task-policy"
    )
    args = parser.parse_args()

    response = requests.post(
        f"{args.url}/baseline",
        json={"episodes": args.episodes},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()

    print("Food Delivery Baseline Report")
    print("=" * 60)
    print(f"Episodes per evaluation: {payload['episodes']}")
    print("=" * 60)

    for row in payload["results"]:
        print(
            f"{row['task_id']:>6} | {row['policy_id']:<8} | score={row['score']:.3f} | "
            f"on_time={row['on_time_rate']:.3f} | avg_delivery={row['avg_delivery_minutes']:.2f}"
        )

    print("=" * 60)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
