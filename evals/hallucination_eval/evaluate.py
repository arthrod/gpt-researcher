"""
Evaluate model outputs for hallucination using the judges library.
"""
import logging
from typing import Dict

from judges.classifiers.hallucination import HaluEvalDocumentSummaryNonFactual

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HallucinationEvaluator:
    """Evaluates model outputs for hallucination using the judges library."""

    def __init__(self, model: str = "openai/gpt-4o"):
        """Initialize the hallucination evaluator."""

        self.summary_judge = HaluEvalDocumentSummaryNonFactual(model=model)

    def evaluate_response(self, model_output: str, source_text: str) -> Dict:
        """
        Evaluate a single model response for hallucination against source documents.
        
        Args:
            model_output: The model's response to evaluate
            source_text: Source text to check summary against
            
        Returns:
            Dict containing evaluation results
        """
        try:
            # Use document summary evaluation
            judgment = self.summary_judge.judge(
                input=source_text,  # The source document
                output=model_output  # The summary to evaluate
            )

            return {
                "output": model_output,
                "source": source_text,
                "score": float(judgment.score) if judgment.score is not None else None,
                "is_hallucination": (
                    judgment.is_hallucination
                    if hasattr(judgment, "is_hallucination")
                    else (float(judgment.score) > 0.5 if judgment.score is not None else None)
                ),
                "reasoning": judgment.reasoning
            }

        except Exception as e:
            logger.error(f"Error evaluating response: {e!s}")
            raise

def main():
    # Example test case
    model_output = "The capital of France is Paris, a city known for its rich history and culture."
    source_text = "Paris is the capital and largest city of France, located in the northern part of the country."

    evaluator = HallucinationEvaluator()
    result = evaluator.evaluate_response(
        model_output=model_output,
        source_text=source_text
    )

    # Print results
    print("\nEvaluation Results:")
    print(f"Output: {result['output']}")
    print(f"Source: {result['source']}")
    print(f"Hallucination: {'Yes' if result['is_hallucination'] else 'No'}")
    print(f"Reasoning: {result['reasoning']}")

if __name__ == "__main__":
    main()