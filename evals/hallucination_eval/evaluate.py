"""
Evaluate model outputs for hallucination using the judges library.
"""
<<<<<<< HEAD

import logging

=======
import logging
from typing import Dict

>>>>>>> 1027e1d0 (Fix linting issues)
from judges.classifiers.hallucination import HaluEvalDocumentSummaryNonFactual

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HallucinationEvaluator:
    """Evaluates model outputs for hallucination using the judges library."""

    def __init__(self, model: str = "openai/gpt-4o"):
        """
        Create a HallucinationEvaluator configured to judge document-summary non-factuality.
        
        Parameters:
            model (str): Identifier of the underlying judge model to use (e.g., "openai/gpt-4o"). Defaults to "openai/gpt-4o".
        """

        self.summary_judge = HaluEvalDocumentSummaryNonFactual(model=model)

<<<<<<< HEAD
    def evaluate_response(self, model_output: str, source_text: str) -> dict:
=======
    def evaluate_response(self, model_output: str, source_text: str) -> Dict:
>>>>>>> 1027e1d0 (Fix linting issues)
        """
<<<<<<< HEAD
        Evaluate a single model response for hallucination against source documents.

        Args:
            model_output: The model's response to evaluate
            source_text: Source text to check summary against

=======
        Evaluate a model-generated summary against source text for hallucination.
        
        Performs a single judgment using the instance's summary_judge and returns a structured result.
        
        Parameters:
            model_output: The model's generated text (summary) to be evaluated.
            source_text: The source document or context to check the summary against.
        
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        Returns:
            dict: {
                "output": model_output,
                "source": source_text,
                "is_hallucination": judgment.score,  # raw score/value returned by the judge (truthy typically indicates hallucination)
                "reasoning": judgment.reasoning      # human-readable explanation from the judge
            }
        
        Exceptions:
            Propagates any exception raised by the underlying judge; callers should handle or allow these to bubble up.
        """
        try:
            # Use document summary evaluation
            judgment = self.summary_judge.judge(
                input=source_text,  # The source document
                output=model_output,  # The summary to evaluate
            )

            return {
                "output": model_output,
                "source": source_text,
                "is_hallucination": judgment.score,
                "reasoning": judgment.reasoning,
            }

        except Exception as e:
            logger.error(f"Error evaluating response: {e!s}")
            raise


def main():
    # Example test case
<<<<<<< HEAD
    model_output = (
        "The capital of France is Paris, a city known for its rich history and culture."
    )
=======
    """
    Run a simple example demonstrating HallucinationEvaluator on a sample model output and source text.
    
    This function constructs a HallucinationEvaluator, evaluates a single model response against a source document,
    and prints the evaluation results (output, source, whether it was classified as a hallucination, and the judge's reasoning).
    Intended as a command-line/demo entry point; it has no return value and produces console output.
    """
    model_output = "The capital of France is Paris, a city known for its rich history and culture."
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
    source_text = "Paris is the capital and largest city of France, located in the northern part of the country."

    evaluator = HallucinationEvaluator()
    result = evaluator.evaluate_response(
        model_output=model_output, source_text=source_text
    )

    # Print results
    print("\nEvaluation Results:")
    print(f"Output: {result['output']}")
    print(f"Source: {result['source']}")
    print(f"Hallucination: {'Yes' if result['is_hallucination'] else 'No'}")
    print(f"Reasoning: {result['reasoning']}")


if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> 1027e1d0 (Fix linting issues)
