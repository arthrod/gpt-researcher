"""
Script to run GPT-Researcher queries and evaluate them for hallucination.
"""

import argparse
import asyncio
import json
import logging
import os
import random

from pathlib import Path

from dotenv import load_dotenv

from gpt_researcher.agent import GPTResearcher
<<<<<<< HEAD
from gpt_researcher.utils.enum import ReportSource, ReportType, Tone
=======
from gpt_researcher.utils.enum import ReportType, ReportSource, Tone
>>>>>>> 1027e1d0 (Fix linting issues)

from .evaluate import HallucinationEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Default paths
DEFAULT_OUTPUT_DIR = "evals/hallucination_eval/results"
DEFAULT_QUERIES_FILE = "evals/hallucination_eval/inputs/search_queries.jsonl"


class ResearchEvaluator:
    """Runs GPT-Researcher queries and evaluates responses for hallucination."""

    def __init__(self, queries_file: str = DEFAULT_QUERIES_FILE):
        """
<<<<<<< HEAD
        Initialize the research evaluator.

        Args:
            queries_file: Path to JSONL file containing search queries
=======
        Create a ResearchEvaluator configured to load queries and run hallucination evaluations.
        
        Parameters:
            queries_file (str): Path to a JSONL file containing search query objects (expects each line to be a JSON object with a "question" key). Defaults to DEFAULT_QUERIES_FILE.
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        """

        self.queries_file = Path(queries_file)
        self.hallucination_evaluator = HallucinationEvaluator()

<<<<<<< HEAD
    def load_queries(self, num_queries: int | None = None) -> list[str]:
=======
    def load_queries(self, num_queries: Optional[int] = None) -> List[str]:
>>>>>>> 1027e1d0 (Fix linting issues)
        """
<<<<<<< HEAD
        Load and optionally sample queries from the JSONL file.

        Args:
            num_queries: Optional number of queries to randomly sample

        Returns:
            List of query strings
=======
        Load queries from the instance's JSONL queries file and optionally return a random sample.
        
        Each line in the file is parsed as JSON and the value under the "question" key is collected in order. If num_queries is provided and is smaller than the total number of queries, a random sample of that many queries is returned; otherwise the full list is returned.
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        """
        queries = []
        with open(self.queries_file) as f:
            for line in f:
                data = json.loads(line.strip())
                queries.append(data["question"])

        if num_queries and num_queries < len(queries):
            return random.sample(queries, num_queries)
        return queries

<<<<<<< HEAD
    async def run_research(self, query: str) -> dict:
=======
    async def run_research(self, query: str) -> Dict:
>>>>>>> 1027e1d0 (Fix linting issues)
        """
<<<<<<< HEAD
        Run a single query through GPT-Researcher.

        Args:
            query: The search query to research

=======
        Run a single search query through GPT-Researcher and return the generated report and raw research context.
        
        Parameters:
            query (str): The search/query string to investigate.
        
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        Returns:
            dict: A mapping with:
                - "query" (str): the original input query.
                - "report" (str): the researcher-generated report in markdown format.
                - "context" (Any): the raw research result/context produced by GPTResearcher.conduct_research().
        """
        researcher = GPTResearcher(
            query=query,
            report_type=ReportType.ResearchReport.value,
            report_format="markdown",
            report_source=ReportSource.Web.value,
            tone=Tone.Objective,
            verbose=True,
        )

        # Run research and get results
        research_result = await researcher.conduct_research()
        report = await researcher.write_report()

        return {
            "query": query,
            "report": report,
            "context": research_result,
        }

    def evaluate_research(
        self, research_data: dict, output_dir: str | None = None
    ) -> dict:
        """
<<<<<<< HEAD
        Evaluate research results for hallucination.

        Args:
            research_data: Dict containing research results and context
            output_dir: Optional directory to save evaluation results

=======
        Evaluate a single research result for hallucination and persist the evaluation.
        
        Evaluates the provided research report against its source context using the hallucination evaluator,
        returns the evaluation result, and appends the result as a JSON line to `evaluation_records.jsonl`
        in the resolved output directory.
        
        Args:
            research_data (Dict): Research result dictionary expected to contain:
                - "query": the original query string
                - "report": the model-generated report to evaluate
                - "context": combined source/context text used for verification (if missing, evaluation is skipped)
            output_dir (Optional[str]): Directory where evaluation records and aggregates are saved.
                If None, defaults to DEFAULT_OUTPUT_DIR.
        
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        Returns:
            Dict: Evaluation result. When "context" is missing, the result will have:
                - "input", "output", "source" (set to "No source text available"),
                - "is_hallucination": None,
                - "confidence_score": None,
                - "reasoning": explanation why evaluation was skipped.
            Otherwise contains the evaluator's output schema.
        
        Side effects:
            - Ensures the output directory exists.
            - Appends the evaluation result as a JSON line to `<output_dir>/evaluation_records.jsonl`.
        """
        # Use default output directory if none provided
        if output_dir is None:
            output_dir = DEFAULT_OUTPUT_DIR

        # Use the final combined context as source text
        source_text = research_data.get("context", "")

        if not source_text:
            logger.warning(
                "No source text found in research results - skipping evaluation"
            )
            eval_result = {
                "input": research_data["query"],
                "output": research_data["report"],
                "source": "No source text available",
                "is_hallucination": None,
                "confidence_score": None,
                "reasoning": "Evaluation skipped - no source text available for verification",
            }
        else:
            # Evaluate the research report for hallucination
            eval_result = self.hallucination_evaluator.evaluate_response(
                model_output=research_data["report"], source_text=source_text
            )

        # Save to output directory
        os.makedirs(output_dir, exist_ok=True)

        # Append to evaluation records
        records_file = Path(output_dir) / "evaluation_records.jsonl"
        with open(records_file, "a") as f:
            f.write(json.dumps(eval_result) + "\n")

        return eval_result


async def main(num_queries: int = 5, output_dir: str = DEFAULT_OUTPUT_DIR):
    """
<<<<<<< HEAD
    Run evaluation on a sample of queries.

    Args:
        num_queries: Number of queries to evaluate
        output_dir: Directory to save results
=======
    Run evaluation across a sampled set of queries: execute research, evaluate for hallucination, save per-query and aggregate results.
    
    This coroutine:
    - Loads up to `num_queries` search queries.
    - For each query, runs GPT-Researcher to produce a report and context, then evaluates the report for hallucination using HallucinationEvaluator.
    - Appends per-query evaluation records to the specified output directory and accumulates aggregate metrics (total responses, evaluated responses, total hallucinated, and hallucination rate).
    - Writes an `aggregate_results.json` file to `output_dir` and prints a short summary to stdout.
    
    Parameters:
        num_queries (int): Number of queries to sample and process (default 5).
        output_dir (str): Directory where per-query records and the aggregate JSON will be written.
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
    """
    evaluator = ResearchEvaluator()

    # Load and sample queries
    queries = evaluator.load_queries(num_queries)
    logger.info(f"Selected {len(queries)} queries for evaluation")

    # Run research and evaluation for each query
    all_results = []
    total_hallucinated = 0
    total_responses = 0
    total_evaluated = 0

    for query in queries:
        try:
            logger.info(f"Processing query: {query}")

            # Run research
            research_data = await evaluator.run_research(query)

            # Evaluate results
            eval_results = evaluator.evaluate_research(
                research_data, output_dir=output_dir
            )

            all_results.append(eval_results)

            # Update counters
            total_responses += 1
            if eval_results["is_hallucination"] is not None:
                total_evaluated += 1
                if eval_results["is_hallucination"]:
                    total_hallucinated += 1

        except Exception as e:
            logger.error(f"Error processing query '{query}': {e!s}")
            continue

    # Calculate hallucination rate
<<<<<<< HEAD
    hallucination_rate = (
        (total_hallucinated / total_evaluated) if total_evaluated > 0 else None
    )
=======
    hallucination_rate = (total_hallucinated / total_evaluated) if total_evaluated > 0 else None
>>>>>>> 1027e1d0 (Fix linting issues)

    # Save aggregate results
    aggregate_results = {
        "total_queries": len(queries),
        "successful_queries": len(all_results),
        "total_responses": total_responses,
        "total_evaluated": total_evaluated,
        "total_hallucinated": total_hallucinated,
        "hallucination_rate": hallucination_rate,
        "results": all_results,
    }

    aggregate_file = Path(output_dir) / "aggregate_results.json"
    with open(aggregate_file, "w") as f:
        json.dump(aggregate_results, f, indent=2)
    logger.info(f"Saved aggregate results to {aggregate_file}")

    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Queries processed: {len(queries)}")
    print(f"Responses evaluated: {total_evaluated}")
    print(f"Responses skipped (no source text): {total_responses - total_evaluated}")
    if hallucination_rate is not None:
        print(f"Hallucination rate: {hallucination_rate * 100:.1f}%")
    else:
        print("No responses could be evaluated due to missing source text")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT-Researcher evaluation")
    parser.add_argument(
        "-n", "--num-queries", type=int, default=5, help="Number of queries to evaluate"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save results",
    )
    args = parser.parse_args()

<<<<<<< HEAD
    asyncio.run(main(args.num_queries, args.output_dir))
=======
    asyncio.run(main(args.num_queries, args.output_dir))
>>>>>>> 1027e1d0 (Fix linting issues)
