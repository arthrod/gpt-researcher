from typing import List
import json
from ..utils.llm import create_chat_completion
from ..actions import stream_output


class SourceCurator:
    """Ranks sources and curates data based on their relevance, credibility and reliability."""

    def __init__(self, researcher):
        self.researcher = researcher

    async def curate_sources(
        self,
        source_data: List,
        max_results: int = 10,
    ) -> List:
        """
        Rank and curate a list of sources using an LLM and return the curated results.
        
        Calls the researcher's prompt to evaluate sources for credibility and relevance, requests the LLM to return a JSON-serializable list of curated sources (up to max_results), and parses that response. On failure (e.g., parsing or LLM error) the original source_data is returned as a fallback. When the researcher is verbose, progress messages are streamed to the researcher's websocket.
        
        Parameters:
            source_data (List): List of source records/documents to be evaluated.
            max_results (int): Maximum number of top sources to request from the LLM (default 10).
        
        Returns:
            List: Curated list of sources (parsed from the LLM's JSON response) or the original source_data on error.
        """
        print(f"\n\nCurating {len(source_data)} sources: {source_data}")
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "research_plan",
                "⚖️ Evaluating and curating sources by credibility and relevance...",
                self.researcher.websocket,
            )

        response = ""
        try:
            response = await create_chat_completion(
                model=self.researcher.cfg.smart_llm_model,
                messages=[
                    {"role": "system", "content": f"{self.researcher.role}"},
                    {"role": "user", "content": self.researcher.prompt_family.curate_sources(
                        self.researcher.query, source_data, max_results)},
                ],
                temperature=0.2,
                max_tokens=8000,
                llm_provider=self.researcher.cfg.smart_llm_provider,
                llm_kwargs=self.researcher.cfg.llm_kwargs,
                cost_callback=self.researcher.add_costs,
            )

            curated_sources = json.loads(response)
            print(f"\n\nFinal Curated sources {len(source_data)} sources: {curated_sources}")

            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "research_plan",
                    f"🏅 Verified and ranked top {len(curated_sources)} most reliable sources",
                    self.researcher.websocket,
                )

            return curated_sources

        except Exception as e:
            print(f"Error in curate_sources from LLM response: {response}")
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "research_plan",
                    f"🚫 Source verification failed: {e!s}",
                    self.researcher.websocket,
                )
            return source_data
