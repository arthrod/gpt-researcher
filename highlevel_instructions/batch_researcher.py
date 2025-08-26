"""
Enhanced batch research manager for multiple research iterations with improved error handling and parallel processing
"""
import asyncio
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ResearchIteration:
    """Data class for research iteration results"""
    instruction: str
    context: Any
    urls: set
    iteration: int
    timestamp: datetime
    duration: float
    error: Optional[str] = None
    success: bool = True


class BatchResearchManager:
    """Enhanced manager for multiple research iterations with improved concurrency and error handling."""

    def __init__(self, researcher, max_concurrent: int = 3):
        """
        Create a BatchResearchManager tied to a researcher and configure concurrency.
        
        Initializes internal storage for iteration results, collected contexts and URLs, sets the concurrency limit and associated asyncio.Semaphore, and prepares a placeholder for an optional progress callback. The `max_concurrent` value controls the maximum number of research iterations that may run concurrently.
        """
        self.researcher = researcher
        self.research_results: List[ResearchIteration] = []
        self.all_contexts = []
        self.all_urls = set()
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.progress_callback: Optional[Callable] = None

    async def conduct_batch_research(
        self,
        research_instructions: List[str],
        parallel: bool = False,
        on_progress: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run a batch of research iterations and return aggregated results and metadata.
        
        If no instructions are provided, performs a single research run and returns its results.
        Sets self.progress_callback to on_progress so per-iteration progress can be reported by internal methods.
        Runs iterations sequentially or in parallel (honoring the manager's concurrency limit) and collects per-iteration ResearchIteration records.
        On completion the manager's researcher.context is updated with a combined context and researcher.visited_urls is updated with all discovered URLs. When the researcher is in verbose mode, start/complete events are streamed.
        
        Parameters:
            research_instructions (List[str]): Instructions/queries for each iteration. An empty list triggers a single-run fallback.
            parallel (bool): If True, execute iterations concurrently (respecting max_concurrent); otherwise run sequentially.
            on_progress (Optional[Callable]): Optional callback invoked for per-iteration progress (stored as self.progress_callback).
        
        Returns:
            Dict[str, Any]: A formatted results dictionary containing per-iteration details and aggregated metadata (contexts, urls, combined_context, timing statistics, counts of successes/failures, and timestamp).
        """
        self.progress_callback = on_progress

        if not research_instructions:
            # Fall back to single research with enhanced tracking
            start_time = datetime.now()
            await self.researcher.research_conductor.conduct_research()
            duration = (datetime.now() - start_time).total_seconds()

            iteration = ResearchIteration(
                instruction=self.researcher.query,
                context=self.researcher.context,
                urls=set(self.researcher.visited_urls),
                iteration=1,
                timestamp=datetime.now(),
                duration=duration
            )

            return self._format_results([iteration], 1)

        if self.researcher.verbose:
            await self._stream_output(
                "batch_research_start",
                f"🔄 Starting {'parallel' if parallel else 'sequential'} batch research with {len(research_instructions)} iterations"
            )

        if parallel:
            results = await self._conduct_parallel_research(research_instructions)
        else:
            results = await self._conduct_sequential_research(research_instructions)

        # Process and combine results
        combined_context = self._combine_contexts_intelligently(results)

        # Update researcher with combined results
        self.researcher.context = combined_context
        self.researcher.visited_urls = self.all_urls

        if self.researcher.verbose:
            successful = sum(1 for r in results if r.success)
            await self._stream_output(
                "batch_research_complete",
                f"🎯 Batch research complete. Successful: {successful}/{len(results)}, "
                f"Total URLs: {len(self.all_urls)}, Combined context: {len(str(combined_context))} chars"
            )

        return self._format_results(results, len(research_instructions))

    async def _conduct_sequential_research(self, instructions: List[str]) -> List[ResearchIteration]:
        """
        Run each research instruction one after another and return their results.
        
        Each instruction is executed by _execute_single_research with a 1-based iteration index. If self.progress_callback is set, it is awaited with (current_iteration, total_iterations, iteration_result) after each iteration. Returns a list of ResearchIteration objects in the same order as the provided instructions.
        
        Exceptions raised by the underlying _execute_single_research are propagated.
        """
        results = []

        for i, instruction in enumerate(instructions, 1):
            result = await self._execute_single_research(instruction, i, len(instructions))
            results.append(result)

            if self.progress_callback:
                await self.progress_callback(i, len(instructions), result)

        return results

    async def _conduct_parallel_research(self, instructions: List[str]) -> List[ResearchIteration]:
        """
        Run multiple research instructions concurrently, respecting the manager's concurrency limit.
        
        Each instruction is executed via the semaphore-wrapped helper (_execute_with_semaphore) so the configured max_concurrent is enforced. This coroutine gathers all tasks and returns a list of ResearchIteration objects in the same order as the provided instructions. If a task raises an exception, it is converted into a failed ResearchIteration containing the exception text in `error` and `success=False`.
         
        Returns:
            List[ResearchIteration]: Per-instruction results (successful iterations or failure records for exceptions), ordered to match `instructions`.
        """
        tasks = []

        for i, instruction in enumerate(instructions, 1):
            task = self._execute_with_semaphore(instruction, i, len(instructions))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ResearchIteration(
                    instruction=instructions[i],
                    context="",
                    urls=set(),
                    iteration=i + 1,
                    timestamp=datetime.now(),
                    duration=0,
                    error=str(result),
                    success=False
                ))
            else:
                processed_results.append(result)

        return processed_results

    async def _execute_with_semaphore(self, instruction: str, iteration: int, total: int) -> ResearchIteration:
        """
        Acquire the concurrency semaphore and run a single research iteration.
        
        This wraps _execute_single_research with the manager's semaphore to enforce the configured concurrency limit.
        
        Parameters:
            instruction (str): Research instruction/query to run.
            iteration (int): Current iteration index (used for reporting).
            total (int): Total number of planned iterations.
        
        Returns:
            ResearchIteration: The result object for this iteration.
        """
        async with self.semaphore:
            return await self._execute_single_research(instruction, iteration, total)

    async def _execute_single_research(self, instruction: str, iteration: int, total: int) -> ResearchIteration:
        """
        Run a single research iteration: set the researcher's query, run the research with a timeout, collect results, and restore the original state.
        
        This coroutine updates the researcher's query to the provided instruction, invokes the research conductor (bounded by a timeout read from `researcher.cfg.research_timeout`, default 120s), and returns a ResearchIteration representing success or failure. On success the returned ResearchIteration contains the iteration's context, discovered URLs, timestamp and duration. On timeout or other exceptions it returns a ResearchIteration with `success=False` and an `error` message. The researcher's original `query` (and original context is preserved in the returned failure records) is always restored before returning. When the researcher is in verbose mode, progress and error events are streamed via _stream_output.
        
        Parameters:
            instruction (str): The query/instruction to run for this iteration.
            iteration (int): 1-based index of this iteration (used for reporting).
            total (int): Total number of iterations in the batch (used for reporting).
        
        Returns:
            ResearchIteration: Result object summarizing the iteration. On failure `success` is False and `error` contains the failure reason.
        """
        start_time = datetime.now()

        if self.researcher.verbose:
            await self._stream_output(
                "batch_iteration",
                f"📚 Research iteration {iteration}/{total}: {instruction[:100]}..."
            )

        # Save original state
        original_query = self.researcher.query
        original_context = self.researcher.context

        try:
            # Update query for this iteration
            self.researcher.query = instruction

            # Conduct research with timeout
            timeout = getattr(self.researcher.cfg, 'research_timeout', 120)
            await asyncio.wait_for(
                self.researcher.research_conductor.conduct_research(),
                timeout=timeout
            )

            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()

            # Create successful iteration result
            result = ResearchIteration(
                instruction=instruction,
                context=self.researcher.context,
                urls=set(self.researcher.visited_urls),
                iteration=iteration,
                timestamp=datetime.now(),
                duration=duration
            )

            # Update aggregated data
            self.all_contexts.append(self.researcher.context)
            self.all_urls.update(self.researcher.visited_urls)

            if self.researcher.verbose:
                await self._stream_output(
                    "iteration_complete",
                    f"✅ Iteration {iteration} complete in {duration:.1f}s. "
                    f"Context: {len(str(self.researcher.context))} chars, URLs: {len(self.researcher.visited_urls)}"
                )

            return result

        except asyncio.TimeoutError:
            error_msg = f"Timeout after {timeout}s"
            logger.error(f"Research iteration {iteration} timed out: {error_msg}")

            return ResearchIteration(
                instruction=instruction,
                context=original_context,
                urls=set(),
                iteration=iteration,
                timestamp=datetime.now(),
                duration=(datetime.now() - start_time).total_seconds(),
                error=error_msg,
                success=False
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in research iteration {iteration}: {error_msg}")

            if self.researcher.verbose:
                await self._stream_output(
                    "iteration_error",
                    f"⚠️ Error in iteration {iteration}: {error_msg[:100]}"
                )

            return ResearchIteration(
                instruction=instruction,
                context=original_context,
                urls=set(),
                iteration=iteration,
                timestamp=datetime.now(),
                duration=(datetime.now() - start_time).total_seconds(),
                error=error_msg,
                success=False
            )

        finally:
            # Always restore original query
            self.researcher.query = original_query

    def _combine_contexts_intelligently(self, results: List[ResearchIteration]) -> str:
        """
        Combine successful iteration contexts into a single deduplicated context string.
        
        This function collects the `context` values from successful ResearchIteration entries and merges them into a single string. Behavior:
        - If no successful contexts are present, returns an empty string.
        - If all successful contexts are strings, each context is split into paragraphs on double-newlines ('\n\n'); paragraphs are stripped, de-duplicated while preserving first-seen order, and joined with a blank-line separator.
        - If contexts are heterogeneous (lists or other types), they are flattened by converting items to strings, stripping, de-duplicating by string value while preserving order, and joined with a blank-line separator.
        
        Returns:
            A combined, deduplicated context string (possibly empty).
        """
        if not results:
            return ""

        # Extract successful contexts
        successful_contexts = [r.context for r in results if r.success and r.context]

        if not successful_contexts:
            return ""

        # Handle different context types
        if all(isinstance(ctx, str) for ctx in successful_contexts):
            # Deduplicate paragraphs/sections
            seen_paragraphs = set()
            combined_parts = []

            for ctx in successful_contexts:
                paragraphs = ctx.split('\n\n')
                for para in paragraphs:
                    para_clean = para.strip()
                    if para_clean and para_clean not in seen_paragraphs:
                        seen_paragraphs.add(para_clean)
                        combined_parts.append(para_clean)

            return "\n\n".join(combined_parts)

        # Handle list contexts
        combined = []
        seen_items = set()

        for ctx in successful_contexts:
            if isinstance(ctx, list):
                for item in ctx:
                    item_str = str(item).strip()
                    if item_str and item_str not in seen_items:
                        seen_items.add(item_str)
                        combined.append(item_str)
            else:
                ctx_str = str(ctx).strip()
                if ctx_str and ctx_str not in seen_items:
                    seen_items.add(ctx_str)
                    combined.append(ctx_str)

        return "\n\n".join(combined)

    def _format_results(self, results: List[ResearchIteration], total_planned: int) -> Dict[str, Any]:
        """
        Build a structured summary of research iterations and aggregate metadata.
        
        Produces a dictionary merging per-iteration data, aggregated statistics, and derived metadata suitable for reporting or downstream consumption.
        
        Parameters:
            results (List[ResearchIteration]): Completed iteration records to include in the output.
            total_planned (int): Number of iterations that were intended/planned for this batch.
        
        Returns:
            Dict[str, Any]: A dictionary with the following keys:
                - contexts: list of contexts from successful iterations.
                - urls: set of all discovered URLs (self.all_urls).
                - iterations_planned: the provided total_planned value.
                - iterations_completed: number of iterations present in `results`.
                - iterations_successful: count of successful iterations.
                - iterations_failed: count of failed iterations.
                - instruction_results: list of per-iteration dicts with keys
                    instruction, context, urls, iteration, timestamp (ISO string),
                    duration, success, and error.
                - combined_context: a single combined context string produced by
                    _combine_contexts_intelligently.
                - metadata: dict containing total_duration, average_duration,
                    total_unique_urls, failed_instructions (list of instruction strings),
                    and timestamp (ISO string when this summary was produced).
        """
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        total_duration = sum(r.duration for r in results)
        avg_duration = total_duration / len(results) if results else 0

        return {
            "contexts": [r.context for r in successful_results],
            "urls": self.all_urls,
            "iterations_planned": total_planned,
            "iterations_completed": len(results),
            "iterations_successful": len(successful_results),
            "iterations_failed": len(failed_results),
            "instruction_results": [
                {
                    "instruction": r.instruction,
                    "context": r.context,
                    "urls": list(r.urls),
                    "iteration": r.iteration,
                    "timestamp": r.timestamp.isoformat(),
                    "duration": r.duration,
                    "success": r.success,
                    "error": r.error
                }
                for r in results
            ],
            "combined_context": self._combine_contexts_intelligently(results),
            "metadata": {
                "total_duration": total_duration,
                "average_duration": avg_duration,
                "total_unique_urls": len(self.all_urls),
                "failed_instructions": [r.instruction for r in failed_results],
                "timestamp": datetime.now().isoformat()
            }
        }

    async def _stream_output(self, event_type: str, message: str):
        """
        Stream a message to the external streaming utility if available.
        
        Attempts to import and call the shared `stream_output` coroutine to send a log/event message
        (using the manager's researcher.websocket). If the `stream_output` import is not available,
        logs the event and message at INFO level instead.
        
        Parameters:
            event_type (str): Short identifier of the event category (e.g., "iteration_start", "error").
            message (str): Human-readable message or payload to stream.
        
        Notes:
            This coroutine does not return a value. ImportError from the optional streaming utility is
            handled internally and will not propagate.
        """
        try:
            from ..actions.utils import stream_output
            await stream_output(
                "logs",
                event_type,
                message,
                self.researcher.websocket,
            )
        except ImportError:
            # Fallback if stream_output is not available
            logger.info(f"{event_type}: {message}")

    def get_iteration_result(self, iteration: int) -> Optional[ResearchIteration]:
        """Get results from a specific iteration."""
        for result in self.research_results:
            if result.iteration == iteration:
                return result
        return None

    def get_successful_results(self) -> List[ResearchIteration]:
        """Get only successful research iterations."""
        return [r for r in self.research_results if r.success]

    def get_failed_results(self) -> List[ResearchIteration]:
        """
        Return all ResearchIteration records marked as failed.
        
        Returns:
            List[ResearchIteration]: Iterations from self.research_results whose `success` is False, in original order.
        """
        return [r for r in self.research_results if not r.success]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Return aggregated statistics about the conducted batch research.
        
        Provides counts, rates, URL and context aggregates, and timing summaries across all recorded iterations.
        If no iterations exist, returns an empty dict.
        
        Returns:
            Dict[str, Any]: A mapping with the following keys:
                - total_iterations (int): Number of iterations recorded.
                - successful (int): Number of iterations marked successful.
                - failed (int): Number of iterations that failed.
                - success_rate (float): successful / total_iterations (0 when no iterations).
                - total_urls_discovered (int): Count of unique URLs collected across all iterations.
                - average_urls_per_iteration (float): total_urls_discovered / successful (0 if no successful iterations).
                - total_context_size (int): Sum of lengths (characters) of successful iterations' contexts.
                - average_context_size (float): Average context length across successful iterations (0 if none).
                - total_duration (float): Sum of durations (seconds) of all iterations.
                - average_duration (float): Mean duration (seconds) per iteration.
        """
        if not self.research_results:
            return {}

        successful = self.get_successful_results()
        failed = self.get_failed_results()

        return {
            "total_iterations": len(self.research_results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.research_results) if self.research_results else 0,
            "total_urls_discovered": len(self.all_urls),
            "average_urls_per_iteration": len(self.all_urls) / len(successful) if successful else 0,
            "total_context_size": sum(len(str(r.context)) for r in successful),
            "average_context_size": sum(len(str(r.context)) for r in successful) / len(successful) if successful else 0,
            "total_duration": sum(r.duration for r in self.research_results),
            "average_duration": sum(r.duration for r in self.research_results) / len(self.research_results)
        }