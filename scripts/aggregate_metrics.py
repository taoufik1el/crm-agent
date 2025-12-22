from typing import Any

model_costs = {
    "gpt-4o-mini-2024-07-18": {"input_tokens": 0.15 / 1e6, "output_tokens": 0.6 / 1e7},
    "gpt-5-mini-2025-08-07": {"input_tokens": 0.25 / 1e6, "output_tokens": 2 / 1e6},
}


def main(input_metrics: dict[str, Any]) -> dict[str, Any]:
    """Aggregate LLM usage metrics from multiple accounts."""
    metrics = {"total_cost": 0.0, "time_taken": 0.0}
    for _, data in input_metrics.items():
        for model_name, model_usage in data["llm_usage"].items():
            metrics["total_cost"] += model_costs[model_name][
                "input_tokens"
            ] * model_usage.get("input_tokens", 0)
        metrics["time_taken"] += data.get("time_taken", 0)
    return metrics


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Aggregate LLM usage metrics")
    parser.add_argument(
        "--input", type=str, required=True, help="Input JSON file with metrics"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file for aggregated metrics",
    )
    args = parser.parse_args()
    with open(args.input) as f:
        input_metrics = json.load(f)
    aggregated_metrics = main(input_metrics)
    with open(args.output, "w") as f:
        json.dump(aggregated_metrics, f, indent=2)
