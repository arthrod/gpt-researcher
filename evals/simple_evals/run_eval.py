import argparse
import asyncio
import json
import os

from collections.abc import Callable
from typing import TypeVar

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from evals.simple_evals.simpleqa_eval import SimpleQAEval
from gpt_researcher.agent import GPTResearcher
from gpt_researcher.utils.enum import ReportSource, ReportType, Tone

# Type variables for generic function
T = TypeVar("T")
R = TypeVar("R")


def map_with_progress(fn: Callable[[T], R], items: list[T]) -> list[R]:
    """Map function over items with progress bar."""
    return [fn(item) for item in tqdm(items)]


# Load environment variables from .env file
load_dotenv()

# Verify all required environment variables
required_env_vars = ["OPENAI_API_KEY", "TAVILY_API_KEY", "LANGCHAIN_API_KEY"]
for var in required_env_vars:
    if not os.getenv(var):
        raise ValueError(f"{var} not found in environment variables")


async def evaluate_single_query(query: str, evaluator: SimpleQAEval) -> dict:
<<<<<<< HEAD
    """
    Run a single query through the researcher pipeline, evaluate the generated report against the ground-truth answer, and return summary metrics.
    
    Performs online research and report generation for `query` using GPTResearcher, looks up the corresponding example in `evaluator.examples`, and grades the generated report with the evaluator. Also prints brief progress and summary lines to stdout.
    
    Returns:
        dict: Summary of the evaluation containing:
            - query (str): The input query/problem text.
            - context_length (int): Length (number of characters) of the retrieved research context.
            - report_length (int): Length (number of characters) of the generated report.
            - cost (float): Monetary cost reported by the researcher for this query.
            - sources (List[str]): List of source URLs used by the researcher.
            - evaluation_score (float): Numeric score returned by the evaluator.
            - evaluation_grade (str): Categorical grade (e.g., "CORRECT", "INCORRECT", "NOT_ATTEMPTED") from the evaluator metrics.
    """
=======
    """Run a single evaluation query and return results"""
>>>>>>> newdev
    print(f"\nEvaluating query: {query}")

    # Run the researcher and get report
    researcher = GPTResearcher(
        query=query,
        report_type=ReportType.ResearchReport.value,
        report_format="markdown",
        report_source=ReportSource.Web.value,
        tone=Tone.Objective,
        verbose=True,
    )
    context = await researcher.conduct_research()
    report = await researcher.write_report()

    # Get the correct answer and evaluate
<<<<<<< HEAD
    example = next(ex for ex in evaluator.examples if ex['problem'] == query)
    correct_answer = example['answer']

    eval_result = evaluator.evaluate_example({
        "problem": query,
        "answer": correct_answer,
        "predicted": report
    })
=======
    example = next(ex for ex in evaluator.examples if ex["problem"] == query)
    correct_answer = example["answer"]

    eval_result = evaluator.evaluate_example(
        {"problem": query, "answer": correct_answer, "predicted": report}
    )
>>>>>>> newdev

    result = {
        "query": query,
        "context_length": len(context),
        "report_length": len(report),
        "cost": researcher.get_costs(),
        "sources": researcher.get_source_urls(),
        "evaluation_score": eval_result["score"],
        "evaluation_grade": eval_result["metrics"]["grade"],
    }

    # Print just the essential info
    print("✓ Completed research and evaluation")
    print(f"  - Sources found: {len(result['sources'])}")
    print(f"  - Evaluation grade: {result['evaluation_grade']}")
    print(f"  - Cost: ${result['cost']:.4f}")

    return result


async def main(num_examples: int):
    """
    Run an end-to-end evaluation loop that runs research-and-evaluation on a set of QA examples.
    
    This async entry point initializes a ChatOpenAI grader and a SimpleQAEval evaluator, iterates over the evaluator's examples,
    runs evaluate_single_query for each example, collects per-query results (including sources, lengths, cost, and evaluation grade/score),
    prints per-query and aggregate statistics (rates, accuracy, F1, and cost summaries), and returns when complete.
    
    Parameters:
        num_examples (int): Number of examples to request from the evaluator. Must be >= 1.
    
    Raises:
        ValueError: If num_examples < 1, if the evaluator loads no examples, or if no results are produced.
        Exception: Re-raises unexpected exceptions after printing a fatal error message.
    
    Side effects:
        - Calls external services (OpenAI via ChatOpenAI and other evaluator/researcher code).
        - Prints progress, warnings, per-query summaries, and aggregate metrics to stdout.
    """
    if num_examples < 1:
        raise ValueError("num_examples must be at least 1")

    try:
        # Initialize the evaluator with specified number of examples
        grader_model = ChatOpenAI(
            temperature=0,
            model_name="gpt-4-turbo",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        evaluator = SimpleQAEval(grader_model=grader_model, num_examples=num_examples)

        if not evaluator.examples:
            raise ValueError("No examples loaded in evaluator")

        print(f"Starting GPT-Researcher evaluation with {num_examples} test queries...")

        results = []
        for example in evaluator.examples:
            if "problem" not in example:
                print(f"Warning: Skipping example without 'problem' key: {example}")
                continue

<<<<<<< HEAD
            query = example['problem']
=======
            query = example["problem"]
>>>>>>> newdev
            print(f"\nEvaluating query: {query}")
            try:
                result = await evaluate_single_query(query, evaluator)
                results.append(result)

                print("✓ Completed research and evaluation")
                print(f"  - Sources found: {len(result['sources'])}")
                print(f"  - Context length: {result['context_length']}")
                print(f"  - Report length: {result['report_length']}")
                print(f"  - Evaluation score: {result['evaluation_score']}")
                print(f"  - Evaluation grade: {result['evaluation_grade']}")
                print(f"  - Cost: ${result['cost']:.4f}")

            except Exception as e:
                print(f"✗ Error evaluating query: {e!s}")
<<<<<<< HEAD
                results.append({
                    'query': query,
                    'error': str(e)
                })
=======
                results.append({"query": query, "error": str(e)})
>>>>>>> newdev

        if not results:
            raise ValueError("No results generated")

        # Print summary for any number of examples
        if num_examples > 0:  # Changed from > 1
            print("\n=== Evaluation Summary ===")
            print(f"Total queries tested: {len(evaluator.examples)}")
            successful = len([r for r in results if "error" not in r])
            print(f"Successful queries: {successful}")
            print(f"Failed queries: {len(evaluator.examples) - successful}")

            if successful > 0:
                # Count the different grades
<<<<<<< HEAD
                correct = sum(1 for r in results if r.get('evaluation_grade') == "CORRECT")
                incorrect = sum(1 for r in results if r.get('evaluation_grade') == "INCORRECT")
                not_attempted = sum(1 for r in results if r.get('evaluation_grade') == "NOT_ATTEMPTED")
=======
                correct = sum(
                    1 for r in results if r.get("evaluation_grade") == "CORRECT"
                )
                incorrect = sum(
                    1 for r in results if r.get("evaluation_grade") == "INCORRECT"
                )
                not_attempted = sum(
                    1 for r in results if r.get("evaluation_grade") == "NOT_ATTEMPTED"
                )
>>>>>>> newdev

                print("\n=== AGGREGATE METRICS ===")
                metrics = {
                    "correct_rate": correct / successful,
                    "incorrect_rate": incorrect / successful,
                    "not_attempted_rate": not_attempted / successful,
                    "answer_rate": (correct + incorrect) / successful,
                }

                # Debug output
                print("\nDebug counts:")
                print(f"Total successful: {successful}")
                print(f"CORRECT: {correct}")
                print(f"INCORRECT: {incorrect}")
                print(f"NOT_ATTEMPTED: {not_attempted}")

                # Calculate accuracy and F1
                metrics["accuracy"] = (
                    correct / (correct + incorrect)  # Accuracy among attempted answers
                    if (correct + incorrect) > 0
                    else 0
                )

                # Precision = correct / attempted
<<<<<<< HEAD
                precision = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0
=======
                precision = (
                    correct / (correct + incorrect) if (correct + incorrect) > 0 else 0
                )
>>>>>>> newdev

                # Recall = correct / total
                recall = correct / successful if successful > 0 else 0

                # F1 = 2 * (precision * recall) / (precision + recall)
                metrics["f1"] = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )

                print(json.dumps(metrics, indent=2))
                print("========================")
                print(f"Accuracy: {metrics['accuracy']:.3f}")
                print(f"F1 Score: {metrics['f1']:.3f}")

                # Print cost metrics
                total_cost = sum(r["cost"] for r in results if "error" not in r)
                print(f"\nTotal cost: ${total_cost:.4f}")
<<<<<<< HEAD
                print(f"Average cost per query: ${total_cost/successful:.4f}")
=======
                print(f"Average cost per query: ${total_cost / successful:.4f}")
>>>>>>> newdev

    except Exception as e:
        print(f"Fatal error in main: {e!s}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT-Researcher evaluation")
    parser.add_argument(
        "--num_examples",
        type=int,
        default=1,
        help="Number of examples to evaluate. Default is 1 example.",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.num_examples))
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
<<<<<<< HEAD
        print(f"Fatal error: {e!s}")
=======
        print(f"Fatal error: {e!s}")
>>>>>>> newdev
