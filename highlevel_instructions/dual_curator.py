"""
Enhanced dual curation system with multi-stage quality assessment and intelligent filtering
"""
<<<<<<< HEAD

import hashlib
import logging

=======
import logging
from typing import List, Any, Dict, Optional
>>>>>>> 1027e1d0 (Fix linting issues)
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CurationStrategy(Enum):
    """Curation strategy types"""

    RELEVANCE_FIRST = "relevance_first"  # Primary: relevance, Secondary: quality
    QUALITY_FIRST = "quality_first"  # Primary: quality, Secondary: relevance
    BALANCED = "balanced"  # Equal weight to both
    STRICT = "strict"  # Must pass both with high scores
    ADAPTIVE = "adaptive"  # Adjust based on source availability


@dataclass
class SourceMetadata:
    """Metadata for source evaluation"""

    source_id: str
    url: str | None = None
    title: str | None = None
    relevance_score: float = 0.0
    quality_score: float = 0.0
    credibility_score: float = 0.0
    freshness_score: float = 0.0
    diversity_score: float = 0.0
    combined_score: float = 0.0
    curation_stage: str = "uncurated"
    timestamp: datetime = field(default_factory=datetime.now)
    rejection_reason: str | None = None
    curator_notes: list[str] = field(default_factory=list)


class DualCuratorManager:
    """Enhanced dual curation manager with comprehensive source assessment."""

<<<<<<< HEAD
    def __init__(
        self, researcher, strategy: CurationStrategy = CurationStrategy.BALANCED
    ):
=======
    def __init__(self, researcher, strategy: CurationStrategy = CurationStrategy.BALANCED):
>>>>>>> 1027e1d0 (Fix linting issues)
        """
<<<<<<< HEAD
        Initialize the dual curator manager.

        Args:
            researcher: The GPTResearcher instance
            strategy: Curation strategy to use
=======
        Initialize the DualCuratorManager and prepare internal curation state.
        
        Creates the manager bound to a researcher and curation strategy, captures the researcher's primary curator, initializes the optional secondary curator slot, and prepares internal structures used across the multi-stage curation flow (history, per-source metadata, and strategy-specific quality thresholds).
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        """
        self.researcher = researcher
        self.primary_curator = researcher.source_curator  # Existing curator
        self.secondary_curator = None
        self.strategy = strategy
        self.curation_history: list[dict] = []
        self.source_metadata: dict[str, SourceMetadata] = {}
        self.quality_thresholds = self._initialize_thresholds()

<<<<<<< HEAD
    def _initialize_thresholds(self) -> dict[str, float]:
=======
    def _initialize_thresholds(self) -> Dict[str, float]:
<<<<<<< HEAD
>>>>>>> 1027e1d0 (Fix linting issues)
        """Initialize quality thresholds based on strategy."""
=======
        """
        Return per-criterion quality thresholds tuned to the current curation strategy.
        
        Thresholds are returned as a mapping with keys:
          - 'relevance': minimum relevance score (0.0–1.0)
          - 'quality': minimum content-quality score
          - 'credibility': minimum source credibility score
          - 'freshness': minimum recency/freshness score
          - 'combined': minimum aggregated combined score used for filtering
        
        Behavior:
          - CurationStrategy.STRICT yields stricter (higher) thresholds.
          - CurationStrategy.BALANCED yields moderate thresholds.
          - Other strategies (e.g., ADAPTIVE or custom) yield lower, more permissive thresholds.
        
        Returns:
          Dict[str, float]: per-criterion threshold values in the range [0.0, 1.0].
        """
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        if self.strategy == CurationStrategy.STRICT:
            return {
                "relevance": 0.7,
                "quality": 0.7,
                "credibility": 0.6,
                "freshness": 0.5,
                "combined": 0.7,
            }
        elif self.strategy == CurationStrategy.BALANCED:
            return {
                "relevance": 0.6,
                "quality": 0.5,
                "credibility": 0.5,
                "freshness": 0.4,
                "combined": 0.55,
            }
        else:  # Adaptive or custom strategies
            return {
                "relevance": 0.5,
                "quality": 0.4,
                "credibility": 0.4,
                "freshness": 0.3,
                "combined": 0.45,
            }

    async def curate_sources(
        self,
<<<<<<< HEAD
        research_data: list[Any],
=======
        research_data: List[Any],
>>>>>>> 1027e1d0 (Fix linting issues)
        min_sources: int = 3,
        max_sources: int | None = None,
        custom_criteria: dict | None = None,
    ) -> list[Any]:
        """
<<<<<<< HEAD
        Apply enhanced dual curation with comprehensive quality assessment.

        Args:
            research_data: Raw research data
            min_sources: Minimum number of sources to retain
            max_sources: Maximum number of sources to retain
            custom_criteria: Custom curation criteria

=======
        Perform a multi-stage, enhanced dual curation of input research sources and return the curated subset.
        
        This coroutine orchestrates four curation stages: primary (relevance filtering), secondary (quality/credibility assessment or external secondary curator), diversity optimization, and final selection according to the manager's strategy. Returns an empty list for empty input. If dual curation is disabled in the researcher configuration or an unexpected error occurs during processing, the method falls back to the single-curation path.
        
        Parameters:
            research_data (List[Any]): Iterable of raw source items to be curated.
            min_sources (int): Minimum number of sources required in the final output; when the active strategy is ADAPTIVE the manager may relax internal thresholds to meet this target.
            max_sources (Optional[int]): Optional hard cap on the number of final selected sources; if provided, the result will be truncated to this size.
            custom_criteria (Optional[Dict]): Optional per-request filtering rules applied at final selection. Recognized keys include:
                - "exclude_domains": iterable of domain strings to remove.
                - "require_keywords": iterable of keywords; retained sources must contain at least one.
        
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        Returns:
            List[Any]: The final curated list of source items (may be shorter than min_sources if constraints and thresholds prevent expansion).
        """
        if not research_data:
            return []

        # Check if dual curation is enabled
        if not getattr(self.researcher.cfg, "dual_curation", False):
            # Single curation fallback
            return await self._single_curation(research_data)

        try:
            # Initialize metadata for all sources
            self._initialize_source_metadata(research_data)

            # Stage 1: Primary curation (relevance-based)
            stage1_results = await self._primary_curation_stage(research_data)

            # Stage 2: Secondary curation (quality-based)
            stage2_results = await self._secondary_curation_stage(stage1_results)

            # Stage 3: Diversity and balance optimization
            stage3_results = await self._diversity_optimization_stage(stage2_results)

            # Stage 4: Final selection based on strategy
            final_results = await self._final_selection_stage(
<<<<<<< HEAD
                stage3_results, min_sources, max_sources, custom_criteria
=======
                stage3_results,
                min_sources,
                max_sources,
                custom_criteria
>>>>>>> 1027e1d0 (Fix linting issues)
            )

            # Log curation statistics
            self._log_curation_statistics(research_data, final_results)

            return final_results

        except Exception as e:
            logger.error(f"Error in enhanced dual curation: {e}")
            # Fallback to simple curation
            return await self._single_curation(research_data)

<<<<<<< HEAD
    async def _single_curation(self, research_data: list[Any]) -> list[Any]:
=======
    async def _single_curation(self, research_data: List[Any]) -> List[Any]:
<<<<<<< HEAD
>>>>>>> 1027e1d0 (Fix linting issues)
        """Fallback single curation when dual curation is disabled or fails."""
=======
        """
        Attempt a single-curator run using the primary curator; on any error, return the first up-to-10 items from the input as a last-resort fallback.
        
        Returns:
            List[Any]: Curated list produced by the primary curator, or the first min(10, len(research_data)) input items if the primary curator call raises an exception.
        """
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        try:
            return await self.primary_curator.curate_sources(research_data)
        except Exception as e:
            print(f"Primary curator failed: {e}")
            # Ultimate fallback - return top N sources
<<<<<<< HEAD
            return research_data[: min(10, len(research_data))]

    async def _primary_curation_stage(self, sources: list[Any]) -> list[Any]:
=======
            return research_data[:min(10, len(research_data))]

    async def _primary_curation_stage(self, sources: List[Any]) -> List[Any]:
>>>>>>> 1027e1d0 (Fix linting issues)
        """
<<<<<<< HEAD
        Primary curation stage focusing on relevance.

        Args:
            sources: Input sources

=======
        Run the primary relevance-focused curation pass and return sources that pass it.
        
        Uses the manager's primary_curator to filter the input sources for relevance. For each returned source, updates the manager's SourceMetadata (if present) by setting a baseline relevance_score (0.7) and curation_stage to "primary_passed". When the researcher is verbose, progress events are streamed.
        
        Parameters:
            sources: Iterable of source items to be evaluated.
        
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        Returns:
            List of sources that passed the primary curation stage.
        """
        if self.researcher.verbose:
            await self._stream_output(
                "first_curation_start",
                f"🔍 Starting primary curation of {len(sources)} sources...",
            )

        # Use existing primary curator
        curated = await self.primary_curator.curate_sources(sources)

        # Update metadata with relevance scores
        for source in curated:
            source_id = self._get_source_id(source)
            if source_id in self.source_metadata:
                # Estimate relevance score based on curation result
                self.source_metadata[
                    source_id
                ].relevance_score = 0.7  # Base score for passing
                self.source_metadata[source_id].curation_stage = "primary_passed"

        if self.researcher.verbose:
            await self._stream_output(
                "first_curation_complete",
                f"✅ Primary curation complete: {len(sources)} → {len(curated)} sources",
            )

        return curated

<<<<<<< HEAD
    async def _secondary_curation_stage(self, sources: list[Any]) -> list[Any]:
=======
    async def _secondary_curation_stage(self, sources: List[Any]) -> List[Any]:
>>>>>>> 1027e1d0 (Fix linting issues)
        """
<<<<<<< HEAD
        Secondary curation stage focusing on quality and credibility.

        Args:
            sources: Sources from primary curation

=======
        Run the secondary curation stage to assess quality and credibility of sources and return the subset that passes.
        
        This will use an externally configured secondary curator if present (via set_secondary_curator) or fall back to the manager's built-in quality assessment. For each source that passes, the function updates that source's SourceMetadata.curation_stage to "secondary_passed". If an empty list is provided, returns an empty list immediately.
        
        Parameters:
            sources (List[Any]): Source items produced by the primary curation stage.
        
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        Returns:
            List[Any]: The sources that passed secondary curation.
        """
        if not sources:
            return []

        if self.researcher.verbose:
            await self._stream_output(
                "second_curation_start",
                f"🎯 Starting secondary quality assessment of {len(sources)} sources...",
            )

        # Apply secondary curator if available
        if self.secondary_curator:
            curated = await self._apply_secondary_curator(sources)
        else:
            # Apply built-in quality assessment
            curated = await self._apply_quality_assessment(sources)

        # Update metadata
        for source in curated:
            source_id = self._get_source_id(source)
            if source_id in self.source_metadata:
                self.source_metadata[source_id].curation_stage = "secondary_passed"

        if self.researcher.verbose:
            await self._stream_output(
                "second_curation_complete",
                f"✅ Secondary curation complete: {len(sources)} → {len(curated)} sources",
            )

        return curated

<<<<<<< HEAD
    async def _apply_secondary_curator(self, sources: list[Any]) -> list[Any]:
=======
    async def _apply_secondary_curator(self, sources: List[Any]) -> List[Any]:
<<<<<<< HEAD
>>>>>>> 1027e1d0 (Fix linting issues)
        """Apply external secondary curator if configured."""
=======
        """
        Apply an external secondary curator to the given sources, falling back to the built-in quality assessment on non-callable configuraton or errors.
        
        If self.secondary_curator exposes a `curate_sources` coroutine it will be awaited; if it is a callable it will be called and awaited. If the configured secondary curator is neither callable nor provides `curate_sources`, or if it raises an exception, this method uses the manager's internal quality assessment routine as a fallback.
        
        Returns:
            List[Any]: The subset of sources accepted by the secondary step (or by the fallback quality assessment).
        """
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        try:
            if hasattr(self.secondary_curator, "curate_sources"):
                return await self.secondary_curator.curate_sources(sources)
            elif callable(self.secondary_curator):
                return await self.secondary_curator(sources)
            else:
                logger.warning(
                    "Secondary curator not callable, using quality assessment"
                )
                return await self._apply_quality_assessment(sources)
        except Exception as e:
            logger.error(f"Secondary curator failed: {e}")
            return await self._apply_quality_assessment(sources)

<<<<<<< HEAD
    async def _apply_quality_assessment(self, sources: list[Any]) -> list[Any]:
=======
    async def _apply_quality_assessment(self, sources: List[Any]) -> List[Any]:
>>>>>>> 1027e1d0 (Fix linting issues)
        """
<<<<<<< HEAD
        Built-in quality assessment for sources.

        Args:
            sources: Sources to assess

=======
        Perform the built-in quality assessment on a list of sources and return those that pass the combined-score threshold.
        
        For each source this:
        - computes quality, credibility, and freshness scores via _calculate_quality_scores,
        - updates the corresponding SourceMetadata (quality_score, credibility_score, freshness_score, combined_score),
        - keeps the source if its combined_score >= self.quality_thresholds['combined'],
        - otherwise records a rejection_reason on the SourceMetadata.
        
        Parameters:
            sources: Iterable of source items to assess.
        
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        Returns:
            List of sources that passed the quality assessment (those retained after threshold filtering).
        """
        assessed_sources = []

        for source in sources:
            scores = await self._calculate_quality_scores(source)
            source_id = self._get_source_id(source)

            if source_id in self.source_metadata:
                metadata = self.source_metadata[source_id]
<<<<<<< HEAD
                metadata.quality_score = scores["quality"]
                metadata.credibility_score = scores["credibility"]
                metadata.freshness_score = scores["freshness"]
=======
                metadata.quality_score = scores['quality']
                metadata.credibility_score = scores['credibility']
                metadata.freshness_score = scores['freshness']
>>>>>>> 1027e1d0 (Fix linting issues)

                # Calculate combined score based on strategy
                metadata.combined_score = self._calculate_combined_score(metadata)

                # Filter based on thresholds
                if metadata.combined_score >= self.quality_thresholds["combined"]:
                    assessed_sources.append(source)
                else:
<<<<<<< HEAD
                    metadata.rejection_reason = (
                        f"Combined score {metadata.combined_score:.2f} below threshold"
                    )

        return assessed_sources

    async def _calculate_quality_scores(self, source: Any) -> dict[str, float]:
=======
                    metadata.rejection_reason = f"Combined score {metadata.combined_score:.2f} below threshold"

        return assessed_sources

    async def _calculate_quality_scores(self, source: Any) -> Dict[str, float]:
<<<<<<< HEAD
>>>>>>> 1027e1d0 (Fix linting issues)
        """Calculate quality scores for a source."""
=======
        """
        Compute heuristic quality, credibility, and freshness scores for a source.
        
        This function uses simple, text-based heuristics on the string representation of `source` to estimate:
        - `quality`: presence of research-oriented phrases (e.g., "study", "journal", "university").
        - `credibility`: presence of authoritative indicators (e.g., ".edu", ".gov", "peer-reviewed").
        - `freshness`: recency inferred by matching the current or recent years in the source text.
        
        Scores are returned as floats in the range [0.0, 1.0]. Default scores of 0.5 are used where no indicators are found or if an error occurs during scoring.
        
        Parameters:
            source: Any
                The source object to evaluate; its string representation is analyzed for heuristic indicators.
        
        Returns:
            Dict[str, float]: A mapping with keys 'quality', 'credibility', and 'freshness', each mapped to a float score between 0.0 and 1.0.
        """
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        scores = {
            "quality": 0.5,  # Default scores
            "credibility": 0.5,
            "freshness": 0.5,
        }

        try:
            # Extract source information
            source_text = str(source)

            # Quality indicators
            quality_indicators = [
                "research",
                "study",
                "analysis",
                "report",
                "journal",
                "university",
                "institute",
                "foundation",
                "organization",
            ]
<<<<<<< HEAD
            quality_count = sum(
                1 for ind in quality_indicators if ind in source_text.lower()
            )
            scores["quality"] = min(1.0, 0.3 + (quality_count * 0.1))
=======
            quality_count = sum(1 for ind in quality_indicators if ind in source_text.lower())
            scores['quality'] = min(1.0, 0.3 + (quality_count * 0.1))
>>>>>>> 1027e1d0 (Fix linting issues)

            # Credibility indicators
            credibility_indicators = [
                ".edu",
                ".gov",
                ".org",
                "peer-reviewed",
                "published",
                "verified",
                "official",
                "authoritative",
            ]
<<<<<<< HEAD
            cred_count = sum(
                1 for ind in credibility_indicators if ind in source_text.lower()
            )
            scores["credibility"] = min(1.0, 0.4 + (cred_count * 0.15))
=======
            cred_count = sum(1 for ind in credibility_indicators if ind in source_text.lower())
            scores['credibility'] = min(1.0, 0.4 + (cred_count * 0.15))
>>>>>>> 1027e1d0 (Fix linting issues)

            # Freshness (simplified - would need actual date parsing)
            current_year = datetime.now().year
            if str(current_year) in source_text or str(current_year - 1) in source_text:
                scores["freshness"] = 0.9
            elif str(current_year - 2) in source_text:
                scores["freshness"] = 0.7
            else:
<<<<<<< HEAD
                scores["freshness"] = 0.5
=======
                scores['freshness'] = 0.5
>>>>>>> 1027e1d0 (Fix linting issues)

        except Exception as e:
            logger.debug(f"Error calculating quality scores: {e}")

        return scores

<<<<<<< HEAD
    async def _diversity_optimization_stage(self, sources: list[Any]) -> list[Any]:
=======
    async def _diversity_optimization_stage(self, sources: List[Any]) -> List[Any]:
>>>>>>> 1027e1d0 (Fix linting issues)
        """
<<<<<<< HEAD
        Optimize source diversity to avoid echo chambers.

        Args:
            sources: Sources from secondary curation

        Returns:
            Diversity-optimized source list
=======
        Optimize and return a more diverse subset of sources to reduce dominance by a single domain or viewpoint.
        
        If there are three or fewer input sources, the list is returned unchanged. Otherwise sources are grouped by domain (via _group_sources_by_domain) and the function selects a small, deterministic number of top sources from each domain group (sources_per_group = max(1, len(sources) // number_of_groups)). The selected sources are returned in aggregated order.
        
        Side effects:
        - Updates SourceMetadata.diversity_score for selected sources in self.source_metadata (score = 1.0 / number_of_groups).
        - Uses helper methods _group_sources_by_domain and _get_source_id; selection is based on group order and original ordering within groups.
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        """
        if len(sources) <= 3:
            return sources  # Too few sources to optimize

        if self.researcher.verbose:
            await self._stream_output(
                "diversity_optimization",
                f"🌈 Optimizing source diversity for {len(sources)} sources...",
            )

        # Group sources by domain/type
        source_groups = self._group_sources_by_domain(sources)

        # Select diverse sources
        diverse_sources = []
<<<<<<< HEAD
        sources_per_group = (
            max(1, len(sources) // len(source_groups)) if source_groups else 1
        )

        for _group_name, group_sources in source_groups.items():
=======
        sources_per_group = max(1, len(sources) // len(source_groups)) if source_groups else 1

        for group_name, group_sources in source_groups.items():
>>>>>>> 1027e1d0 (Fix linting issues)
            # Take top sources from each group
            selected = group_sources[:sources_per_group]
            diverse_sources.extend(selected)

            # Update diversity scores
            for source in selected:
                source_id = self._get_source_id(source)
                if source_id in self.source_metadata:
                    # Higher diversity score for sources from smaller groups
<<<<<<< HEAD
                    self.source_metadata[source_id].diversity_score = 1.0 / len(
                        source_groups
                    )

        return diverse_sources

    def _group_sources_by_domain(self, sources: list[Any]) -> dict[str, list[Any]]:
=======
                    self.source_metadata[source_id].diversity_score = 1.0 / len(source_groups)

        return diverse_sources

    def _group_sources_by_domain(self, sources: List[Any]) -> Dict[str, List[Any]]:
<<<<<<< HEAD
>>>>>>> 1027e1d0 (Fix linting issues)
        """Group sources by domain or type."""
=======
        """
        Group sources into coarse domain categories based on simple heuristics applied to each source's string representation.
        
        This function inspects each source (converted to a lower-case string) and assigns it to one of these domains: "academic", "government", "news", "blog", "organization", or "general". The classification is heuristic and keyword-based (for example, occurrences of ".edu", ".gov", "news", "blog", ".org", etc.) and is intended for coarse-grained grouping rather than authoritative domain detection.
        
        Parameters:
            sources (List[Any]): Iterable of source items (strings or objects); each item is converted to a string for matching.
        
        Returns:
            Dict[str, List[Any]]: Mapping from domain name to the list of original source items assigned to that domain.
        """
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        groups = {}

        for source in sources:
            # Simplified domain extraction
            domain = "general"
            source_str = str(source).lower()

<<<<<<< HEAD
            if any(term in source_str for term in [".edu", "university", "academic"]):
=======
            if any(term in source_str for term in ['.edu', 'university', 'academic']):
>>>>>>> 1027e1d0 (Fix linting issues)
                domain = "academic"
            elif any(term in source_str for term in [".gov", "government", "official"]):
                domain = "government"
            elif any(term in source_str for term in ["news", "media", "press"]):
                domain = "news"
            elif any(term in source_str for term in ["blog", "medium", "substack"]):
                domain = "blog"
            elif any(
                term in source_str for term in [".org", "organization", "foundation"]
            ):
                domain = "organization"

            if domain not in groups:
                groups[domain] = []
            groups[domain].append(source)

        return groups

    async def _final_selection_stage(
        self,
        sources: list[Any],
        min_sources: int,
        max_sources: int | None,
        custom_criteria: dict | None,
    ) -> list[Any]:
        """
<<<<<<< HEAD
        Final selection stage based on strategy and constraints.

        Args:
            sources: Sources from diversity optimization
            min_sources: Minimum sources required
            max_sources: Maximum sources allowed
            custom_criteria: Custom selection criteria

=======
        Perform the final selection of curated sources according to the active strategy, size constraints, and any custom criteria.
        
        This stage:
        - Ranks the provided sources by their stored `combined_score` (from SourceMetadata) and selects in descending order.
        - If the selection has fewer than `min_sources` and the strategy is ADAPTIVE, triggers an asynchronous threshold relaxation to attempt to broaden the candidate pool.
        - Enforces `max_sources` by truncating the ranked list when provided.
        - Applies `custom_criteria` filtering (e.g., `exclude_domains`, `require_keywords`) if supplied.
        - Marks chosen sources' metadata curation_stage as "final_selected".
        
        Parameters:
        - sources: List of candidate sources (expected to have corresponding entries in self.source_metadata).
        - min_sources: Minimum number of sources desired; used to decide whether to relax thresholds under ADAPTIVE strategy.
        - max_sources: Optional upper bound on the number of returned sources.
        - custom_criteria: Optional dict of user-specified filters passed to _apply_custom_criteria (supported keys include `exclude_domains` and `require_keywords`).
        
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        Returns:
        - A list of selected sources (possibly empty). Exceptions are not documented here; errors in upstream stages are handled elsewhere.
        """
        if not sources:
            return []

        # Sort sources by combined score
        scored_sources = []
        for source in sources:
            source_id = self._get_source_id(source)
            if source_id in self.source_metadata:
                score = self.source_metadata[source_id].combined_score
                scored_sources.append((score, source))

        scored_sources.sort(reverse=True, key=lambda x: x[0])

        # Apply constraints
        selected = [source for _, source in scored_sources]

        # Ensure minimum sources
        if len(selected) < min_sources and self.strategy == CurationStrategy.ADAPTIVE:
            # Relax thresholds if needed
            await self._relax_thresholds_and_reselect(sources, min_sources)

        # Apply maximum constraint
        if max_sources and len(selected) > max_sources:
            selected = selected[:max_sources]

        # Apply custom criteria if provided
        if custom_criteria:
            selected = self._apply_custom_criteria(selected, custom_criteria)

        # Update final metadata
        for source in selected:
            source_id = self._get_source_id(source)
            if source_id in self.source_metadata:
                self.source_metadata[source_id].curation_stage = "final_selected"

        return selected

<<<<<<< HEAD
    async def _relax_thresholds_and_reselect(
        self, sources: list[Any], min_sources: int
    ):
=======
    async def _relax_thresholds_and_reselect(self, sources: List[Any], min_sources: int):
<<<<<<< HEAD
>>>>>>> 1027e1d0 (Fix linting issues)
        """Relax quality thresholds if minimum sources not met."""
=======
        """
        Relax the manager's quality thresholds to try to meet a minimum required number of sources.
        
        This mutates self.quality_thresholds in-place (each threshold is multiplied by 0.8) to relax selection criteria by 20%. The method is intended to be followed by a re-evaluation/reselection pass using the updated thresholds; it does not itself perform that re-curation.
        
        Parameters:
            sources (List[Any]): Candidate sources that would be re-evaluated after thresholds are relaxed.
            min_sources (int): The target minimum number of sources that the relaxed thresholds aim to achieve.
        
        Returns:
            None
        """
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        logger.info("Relaxing thresholds to meet minimum source requirement")

        # Reduce all thresholds by 20%
        for key in self.quality_thresholds:
            self.quality_thresholds[key] *= 0.8

        # Re-evaluate sources with relaxed thresholds
        # This would trigger a partial re-curation with new thresholds

<<<<<<< HEAD
    def _apply_custom_criteria(self, sources: list[Any], criteria: dict) -> list[Any]:
        """Apply custom filtering criteria."""
        filtered = sources

        if "exclude_domains" in criteria:
            excluded = criteria["exclude_domains"]
            filtered = [s for s in filtered if not any(d in str(s) for d in excluded)]

        if "require_keywords" in criteria:
            keywords = criteria["require_keywords"]
            filtered = [
                s for s in filtered if any(k in str(s).lower() for k in keywords)
            ]
=======
    def _apply_custom_criteria(self, sources: List[Any], criteria: Dict) -> List[Any]:
        """
        Filter a list of sources using simple, user-provided criteria.
        
        This applies two optional criteria keys from `criteria`:
        - "exclude_domains": an iterable of domain substrings; any source whose string representation contains any of these substrings is removed (substring match, case-sensitive).
        - "require_keywords": an iterable of keywords; a source is kept only if any keyword appears in the lowercased string representation of the source (the source is lowercased before matching — provide keywords in lowercase to ensure expected matches).
        
        Parameters:
            sources (List[Any]): Iterable of source objects (their string representation is used for matching).
            criteria (Dict): Filtering rules; supported keys are `'exclude_domains'` and `'require_keywords'`.
        
        Returns:
            List[Any]: The filtered list of sources.
        """
        filtered = sources

        if 'exclude_domains' in criteria:
            excluded = criteria['exclude_domains']
            filtered = [s for s in filtered if not any(d in str(s) for d in excluded)]

        if 'require_keywords' in criteria:
            keywords = criteria['require_keywords']
            filtered = [s for s in filtered if any(k in str(s).lower() for k in keywords)]
>>>>>>> 1027e1d0 (Fix linting issues)

        return filtered

    def _calculate_combined_score(self, metadata: SourceMetadata) -> float:
        """
        Compute a weighted combined score (clamped to 1.0) for a source according to the active CurationStrategy.
        
        Weights are applied to the SourceMetadata fields relevance_score, quality_score, credibility_score, and freshness_score and vary by strategy:
        - RELEVANCE_FIRST: higher weight on relevance.
        - QUALITY_FIRST: higher weight on quality and credibility.
        - BALANCED or other: balanced weights across factors.
        
        If metadata.diversity_score > 0, a small diversity bonus (10% of diversity_score) is added before clamping.
        
        Parameters:
            metadata (SourceMetadata): Source metadata containing relevance_score, quality_score, credibility_score, freshness_score, and diversity_score.
        
        Returns:
            float: Combined score in the range [0.0, 1.0].
        """
        if self.strategy == CurationStrategy.RELEVANCE_FIRST:
            weights = {
                "relevance": 0.5,
                "quality": 0.2,
                "credibility": 0.2,
                "freshness": 0.1,
            }
        elif self.strategy == CurationStrategy.QUALITY_FIRST:
            weights = {
                "relevance": 0.2,
                "quality": 0.4,
                "credibility": 0.3,
                "freshness": 0.1,
            }
        else:  # BALANCED or other
            weights = {
                "relevance": 0.3,
                "quality": 0.3,
                "credibility": 0.25,
                "freshness": 0.15,
            }

        score = (
            weights["relevance"] * metadata.relevance_score
            + weights["quality"] * metadata.quality_score
            + weights["credibility"] * metadata.credibility_score
            + weights["freshness"] * metadata.freshness_score
        )

        # Add diversity bonus if available
        if metadata.diversity_score > 0:
            score += metadata.diversity_score * 0.1

        return min(1.0, score)

<<<<<<< HEAD
    def _initialize_source_metadata(self, sources: list[Any]):
=======
    def _initialize_source_metadata(self, sources: List[Any]):
<<<<<<< HEAD
>>>>>>> 1027e1d0 (Fix linting issues)
        """Initialize metadata for all sources."""
=======
        """
        Ensure SourceMetadata exists for each source in `sources`.
        
        For each item in `sources` this creates and stores a SourceMetadata entry keyed by a stable source_id (generated via _get_source_id) when one does not already exist. New entries are initialized with the extracted url and title (via _extract_url and _extract_title). Existing metadata entries are left unchanged.
        
        Parameters:
            sources (List[Any]): Iterable of raw source objects/strings to register in source_metadata.
        
        Side effects:
            Mutates self.source_metadata by adding SourceMetadata instances for previously unseen sources.
        """
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        for source in sources:
            source_id = self._get_source_id(source)
            if source_id not in self.source_metadata:
                self.source_metadata[source_id] = SourceMetadata(
                    source_id=source_id,
                    url=self._extract_url(source),
                    title=self._extract_title(source),
                )

    def _get_source_id(self, source: Any) -> str:
        """
        Generate a deterministic short identifier for a source.
        
        Returns a 16-hex-digit string derived from the MD5 hash of the source's string representation. The ID is stable for the same source input but not cryptographically collision-proof; collisions are possible due to MD5 and truncation.
        """
        source_str = str(source)
        return hashlib.md5(source_str.encode()).hexdigest()[:16]

<<<<<<< HEAD
    def _extract_url(self, source: Any) -> str | None:
=======
    def _extract_url(self, source: Any) -> Optional[str]:
<<<<<<< HEAD
>>>>>>> 1027e1d0 (Fix linting issues)
        """Extract URL from source if available."""
=======
        """
        Return the first URL-like substring found in the given source's string representation.
        
        This performs a simple, heuristic extraction by converting `source` to a string, locating the first occurrence of the substring starting with "http", and returning characters from that position up to the next space (or end of string). Does not validate or normalize the URL and may return partial or malformed results; returns None if no "http" substring is present.
        """
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        source_str = str(source)
        # Simplified URL extraction
        if "http" in source_str:
            start = source_str.find("http")
            end = source_str.find(" ", start)
            if end == -1:
                end = len(source_str)
            return source_str[start:end]
        return None

<<<<<<< HEAD
    def _extract_title(self, source: Any) -> str | None:
=======
    def _extract_title(self, source: Any) -> Optional[str]:
<<<<<<< HEAD
>>>>>>> 1027e1d0 (Fix linting issues)
        """Extract title from source if available."""
=======
        """
        Attempt to extract a human-readable title from a source object.
        
        This is a placeholder implementation that currently returns None. Implementers should extract a title when present (for example from dict keys like 'title' or 'name', object attributes like .title, or by deriving a title from the source URL/content) and return it as a string.
        
        Parameters:
            source (Any): The source item to inspect; structure may be a dict, object, or raw string/URL.
        
        Returns:
            Optional[str]: The extracted title if available, otherwise None.
        """
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        # This would need proper implementation based on source structure
        return None

<<<<<<< HEAD
    def _log_curation_statistics(self, original: list[Any], final: list[Any]):
=======
    def _log_curation_statistics(self, original: List[Any], final: List[Any]):
<<<<<<< HEAD
>>>>>>> 1027e1d0 (Fix linting issues)
        """Log curation statistics for analysis."""
=======
        """
        Record metrics about a curation run and append them to the manager's history.
        
        Builds a statistics record (timestamp, original/final counts, reduction rate, active strategy, and per-stage metrics),
        appends it to self.curation_history, and emits an info-level summary when the researcher is in verbose mode.
        
        Parameters:
            original (List[Any]): The list of sources before curation.
            final (List[Any]): The list of sources after curation (final selection).
        """
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        stats = {
            "timestamp": datetime.now().isoformat(),
            "original_count": len(original),
            "final_count": len(final),
            "reduction_rate": 1 - (len(final) / len(original)) if original else 0,
            "strategy": self.strategy.value,
            "stage_metrics": self._calculate_stage_metrics(),
        }

        self.curation_history.append(stats)

        if self.researcher.verbose:
<<<<<<< HEAD
            logger.info(
                f"Curation complete: {stats['original_count']} → {stats['final_count']} sources "
                f"({stats['reduction_rate']:.1%} reduction)"
            )

    def _calculate_stage_metrics(self) -> dict[str, int]:
=======
            logger.info(f"Curation complete: {stats['original_count']} → {stats['final_count']} sources "
                       f"({stats['reduction_rate']:.1%} reduction)")

    def _calculate_stage_metrics(self) -> Dict[str, int]:
<<<<<<< HEAD
>>>>>>> 1027e1d0 (Fix linting issues)
        """Calculate metrics for each curation stage."""
=======
        """
        Return counts of sources in each curation stage.
        
        Iterates over the manager's SourceMetadata entries and tallies how many sources are in each of the tracked stages:
        'uncurated', 'primary_passed', 'secondary_passed', and 'final_selected'. Any metadata with an unexpected
        stage value is ignored.
        
        Returns:
            Dict[str, int]: Mapping from stage name to count for the four tracked stages.
        """
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        metrics = {
            "uncurated": 0,
            "primary_passed": 0,
            "secondary_passed": 0,
            "final_selected": 0,
        }

        for metadata in self.source_metadata.values():
            stage = metadata.curation_stage
            if stage in metrics:
                metrics[stage] += 1

        return metrics

    async def _stream_output(self, event_type: str, message: str):
        """
        Stream a textual event to the researcher's websocket (if available) and optionally log it.
        
        If the researcher is running in verbose mode the message is logged locally. If the researcher
        has a `websocket` attribute and it is set, this coroutine will attempt to stream the event
        via the `stream_output` utility. ImportError raised while importing the streaming helper is
        silently ignored so calling code is not affected when the optional streaming dependency is missing.
        
        Parameters:
            event_type (str): Short event category/name used by the streaming protocol (e.g., "info", "progress", "error").
            message (str): Human-readable message payload to log and/or stream.
        
        Returns:
            None
        """
        if self.researcher.verbose:
            logger.info(message)

<<<<<<< HEAD
        if hasattr(self.researcher, "websocket") and self.researcher.websocket:
=======
        if hasattr(self.researcher, 'websocket') and self.researcher.websocket:
>>>>>>> 1027e1d0 (Fix linting issues)
            try:
<<<<<<< HEAD
                from ..actions.utils import stream_output

=======
                from gpt_researcher.actions.utils import stream_output
>>>>>>> 1f1be9a8 (updating)
                await stream_output(
                    "logs", event_type, message, self.researcher.websocket
                )
            except ImportError:
                pass

    def set_secondary_curator(self, curator_agent):
        """
        Set the secondary curator agent used for dual-curation flows.
        
        If the researcher configuration exposes a `secondary_curator_agent` attribute, this method mirrors the provided agent into that config for persistence/visibility.
        
        Parameters:
            curator_agent: The curator agent (callable or object) to assign as the secondary curator.
        """
        self.secondary_curator = curator_agent
        if hasattr(self.researcher.cfg, "secondary_curator_agent"):
            self.researcher.cfg.secondary_curator_agent = curator_agent

    def set_strategy(self, strategy: CurationStrategy):
        """
        Set the active curation strategy and reinitialize per-criterion quality thresholds.
        
        This updates the manager's strategy to `strategy` and recomputes quality thresholds
        (via _initialize_thresholds) so subsequent curation runs use the new strategy's
        thresholds.
        """
        self.strategy = strategy
        self.quality_thresholds = self._initialize_thresholds()

<<<<<<< HEAD
    def get_curation_report(self) -> dict[str, Any]:
=======
    def get_curation_report(self) -> Dict[str, Any]:
<<<<<<< HEAD
>>>>>>> 1027e1d0 (Fix linting issues)
        """Get comprehensive curation report."""
=======
        """
        Return a comprehensive report of the current curation state.
        
        The report is a dictionary with the following keys:
        - total_sources_processed (int): number of sources tracked in metadata.
        - stage_metrics (Dict[str, int]): counts of sources per curation stage (e.g. 'uncurated', 'primary_passed', 'secondary_passed', 'final_selected').
        - average_scores (Dict[str, float]): average numeric scores across tracked sources (relevance, quality, credibility, freshness, combined).
        - rejection_reasons (Dict[str, int]): aggregated counts of rejection reason categories.
        - curation_history (List[Any]): list of recent curation history entries (up to the last 10).
        
        Returns:
            Dict[str, Any]: The aggregated curation report described above.
        """
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        return {
            "total_sources_processed": len(self.source_metadata),
            "stage_metrics": self._calculate_stage_metrics(),
            "average_scores": self._calculate_average_scores(),
            "rejection_reasons": self._get_rejection_summary(),
            "curation_history": self.curation_history[-10:],  # Last 10 curations
        }

<<<<<<< HEAD
    def _calculate_average_scores(self) -> dict[str, float]:
=======
    def _calculate_average_scores(self) -> Dict[str, float]:
<<<<<<< HEAD
>>>>>>> 1027e1d0 (Fix linting issues)
        """Calculate average scores across all sources."""
=======
        """
        Compute average relevance, quality, credibility, freshness, and combined scores across all tracked sources.
        
        Returns:
            A dict mapping score names to their average value (keys: 'relevance', 'quality', 'credibility', 'freshness', 'combined').
            Returns an empty dict if no source metadata is available.
        """
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        if not self.source_metadata:
            return {}

        totals = {
            "relevance": 0,
            "quality": 0,
            "credibility": 0,
            "freshness": 0,
            "combined": 0,
        }

        count = len(self.source_metadata)
        for metadata in self.source_metadata.values():
<<<<<<< HEAD
            totals["relevance"] += metadata.relevance_score
            totals["quality"] += metadata.quality_score
            totals["credibility"] += metadata.credibility_score
            totals["freshness"] += metadata.freshness_score
            totals["combined"] += metadata.combined_score

        return {k: v / count for k, v in totals.items()}

    def _get_rejection_summary(self) -> dict[str, int]:
=======
            totals['relevance'] += metadata.relevance_score
            totals['quality'] += metadata.quality_score
            totals['credibility'] += metadata.credibility_score
            totals['freshness'] += metadata.freshness_score
            totals['combined'] += metadata.combined_score

        return {k: v / count for k, v in totals.items()}

    def _get_rejection_summary(self) -> Dict[str, int]:
<<<<<<< HEAD
>>>>>>> 1027e1d0 (Fix linting issues)
        """Get summary of rejection reasons."""
=======
        """
        Return a mapping of rejection reason categories to their occurrence counts.
        
        Iterates over stored SourceMetadata entries and, for each non-empty rejection_reason,
        splits the string at the first ':' to derive a reason category and tallies how many
        sources were rejected for that category.
        
        Returns:
            Dict[str, int]: Keys are rejection reason categories (substring before the first ':'),
            values are counts of sources with that category.
        """
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        reasons = {}
        for metadata in self.source_metadata.values():
            if metadata.rejection_reason:
                reason = metadata.rejection_reason.split(":")[0]  # Get reason category
                reasons[reason] = reasons.get(reason, 0) + 1
        return reasons
