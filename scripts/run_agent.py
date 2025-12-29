import json
import logging
import re
import time
from pathlib import Path

from langchain_core.callbacks import get_usage_metadata_callback
from tqdm import tqdm

from agent.graph import create_agent_graph
from agent.main import run_agent
from mcp_server.server import DATA_DIR

question_map = [
    "What is the maximum budget that Marcus, the CFO, has set for operational efficiency tools this fiscal year?",
    "What is the approved budget amount for the order once the DPA and security review are signed off?",
    "What date and time is scheduled for Bertrand’s visio meeting to review the onboarding and training plan?",
    "By what time does Herr Hagedorn need to receive the PDF invoice to ensure it is included in this month’s payment run?",
    "What date and time is the wider stakeholder meeting confirmed for?",
    "When will the NexiFlow system switch from the calibration phase to active monitoring and start sending real alerts?",
    "What is the total contract value (TCV) for the Trans-Pacific route expansion after finalizing the $45k pricing?",
    "how many calls have been made with this account",
    "have they raised concerns about pricing or contract terms",
    "what is the principal factors that influence the decision process for this account",
]


def run_agent_on_all_accounts(baseline_mode: bool, output_path: str) -> None:
    """Run the agent on all account data files in the DATA_DIR."""
    output_json_path = Path(output_path)
    agent = create_agent_graph(streaming=False)
    if output_json_path.exists():
        with output_json_path.open("r") as f:
            results = json.load(f)
    else:
        results = {}
    for e, file in tqdm(enumerate(DATA_DIR.iterdir())):
        # get id from filename: like account_1.json with regex
        match = re.match(r"account_(\d+)\.json", file.name)
        if not match:
            continue
        account_id = int(match.group(1))
        if str(account_id) in results:
            continue
        question = question_map[account_id - 1]
        start = time.time()
        try:
            with get_usage_metadata_callback() as usage_cb:
                result = run_agent(
                    agent, question, account_id=account_id, baseline=baseline_mode
                )
                end = time.time()
                llm_usage = usage_cb.usage_metadata
            results[account_id] = {
                "question": question,
                "response": result,
                "llm_usage": llm_usage,
                "time_taken": end - start,
            }
        except Exception as e:
            logging.log(logging.ERROR, f"Error processing account {account_id}: {e}")
    with output_json_path.open("w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the ML Engineer Agent on all accounts"
    )
    parser.add_argument(
        "--baseline", action="store_true", help="Run the agent in baseline mode"
    )
    # output json path
    parser.add_argument(
        "--output", type=str, default="agent_results.json", help="Output JSON file path"
    )
    args = parser.parse_args()
    run_agent_on_all_accounts(baseline_mode=args.baseline, output_path=args.output)
