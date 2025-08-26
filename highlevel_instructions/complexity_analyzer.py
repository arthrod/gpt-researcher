"""
Enhanced complexity analyzer with multi-dimensional analysis and adaptive recommendations
"""

import asyncio
import json
import logging
<<<<<<< HEAD

from dataclasses import asdict, dataclass
=======
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
>>>>>>> 1027e1d0 (Fix linting issues)
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ComplexityDimension(Enum):
    """Enumeration of complexity dimensions"""

    CONTROVERSY = "controversy_level"
    PERSPECTIVES = "multiple_perspectives"
    LEGAL_CERTAINTY = "legal_certainty"
    NOVELTY = "novelty"
    TECHNICAL = "technical_complexity"
    SCOPE = "scope_breadth"
    DATA_AVAILABILITY = "data_availability"
    INTERDISCIPLINARY = "interdisciplinary_nature"
    TEMPORAL = "temporal_relevance"
    REGULATORY = "regulatory_complexity"


@dataclass
class ComplexityMetrics:
    """Data class for complexity metrics"""

    controversy_level: float = 5.0
    multiple_perspectives: float = 5.0
    legal_certainty: float = 5.0
    novelty: float = 5.0
    technical_complexity: float = 5.0
    scope_breadth: float = 5.0
    data_availability: float = 5.0
    interdisciplinary_nature: float = 5.0
    temporal_relevance: float = 5.0
    regulatory_complexity: float = 5.0
    overall_complexity: float = 5.0
    confidence_score: float = 0.5


@dataclass
class ResearchRecommendations:
    """Data class for research recommendations"""

    recommended_min_words: int
    recommended_max_words: int
    suggested_iterations: int
    special_attention_areas: list[str]
    recommended_report_type: str
    research_depth: str
    source_diversity_needed: str
    fact_checking_rigor: str
    rationale: str


class ComplexityAnalyzer:
    """Enhanced analyzer for topic complexity with adaptive recommendations."""

    def __init__(self, config):
        """
<<<<<<< HEAD
        Initialize the complexity analyzer.

        Args:
            config: Configuration object
=======
        Create a ComplexityAnalyzer instance.
        
        Initializes the analyzer with the provided configuration, sets up an empty in-memory cache for results, and creates an analysis history list.
        
        Parameters:
            config: Configuration object containing analyzer settings (e.g., LLM provider configs, complexity_factors flags, and default word-counts) used by other methods in this class.
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        """
        self.config = config
        self.cache = {}
        self.analysis_history = []

    async def analyze_complexity(
        self,
        query: str,
<<<<<<< HEAD
        context: list[str] | None = None,
=======
        context: Optional[List[str]] = None,
>>>>>>> 1027e1d0 (Fix linting issues)
        force_analysis: bool = False,
        include_comparative: bool = True,
    ) -> dict[str, Any]:
        """
<<<<<<< HEAD
        Perform enhanced complexity analysis with caching and comparative analysis.

        Args:
            query: The research query
            context: Optional research context
            force_analysis: Force new analysis even if cached
            include_comparative: Include comparative complexity analysis

=======
        Perform a multi-stage complexity analysis of a research query, returning structured scores and adaptive recommendations.
        
        This orchestrates three asynchronous analyses (primary complexity scoring, optional domain-specific classification, and feasibility estimation), merges their outputs, caches the combined result, and records a history entry. If a cached result exists and force_analysis is False, the cached result is returned (it will include a 'from_cache' flag). If complexity analysis is disabled or an error occurs, a safe default complexity structure is returned.
        
        Parameters:
            query (str): The research query or prompt to analyze.
            context (Optional[List[str]]): Optional contextual snippets to include in prompts (used for richer primary analysis and cache key generation).
            force_analysis (bool): When True, bypasses the cache and forces recomputation.
            include_comparative (bool): When True, runs domain-specific comparative analysis; otherwise a minimal domain skeleton is used.
        
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        Returns:
            Dict[str, Any]: A merged analysis containing per-dimension scores, overall_complexity, confidence_score, adaptive ResearchRecommendations, domain_analysis, feasibility_analysis, metadata (including analysis_timestamp and analysis_version), and caching metadata ('from_cache' when applicable).
        """
        # Check cache unless forced
        cache_key = self._generate_cache_key(query, context)
        if not force_analysis and cache_key in self.cache:
            cached_result = self.cache[cache_key]
            cached_result["from_cache"] = True
            return cached_result

        # Check if complexity analysis is enabled
        if not self._is_complexity_analysis_enabled():
            return self._get_default_complexity()

        try:
            # Perform parallel analyses for comprehensive assessment
            analyses = await asyncio.gather(
                self._analyze_primary_complexity(query, context),
                self._analyze_domain_specifics(query)
                if include_comparative
                else self._get_empty_domain_analysis(),
                self._analyze_research_feasibility(query, context),
                return_exceptions=True,
            )

            # Process results
<<<<<<< HEAD
            primary_analysis = (
                analyses[0] if not isinstance(analyses[0], Exception) else {}
            )
            domain_analysis = (
                analyses[1] if not isinstance(analyses[1], Exception) else {}
            )
            feasibility_analysis = (
                analyses[2] if not isinstance(analyses[2], Exception) else {}
            )
=======
            primary_analysis = analyses[0] if not isinstance(analyses[0], Exception) else {}
            domain_analysis = analyses[1] if not isinstance(analyses[1], Exception) else {}
            feasibility_analysis = analyses[2] if not isinstance(analyses[2], Exception) else {}
>>>>>>> 1027e1d0 (Fix linting issues)

            # Combine analyses
            combined_result = self._combine_analyses(
                primary_analysis, domain_analysis, feasibility_analysis, query
            )

            # Cache the result
            self.cache[cache_key] = combined_result
            self.analysis_history.append({
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "complexity": combined_result.get("overall_complexity", 5),
            })

            return combined_result

        except Exception as e:
            logger.error(f"Error in enhanced complexity analysis: {e}")
            return self._get_default_complexity()

<<<<<<< HEAD
    async def _analyze_primary_complexity(
        self, query: str, context: list[str] | None
    ) -> dict[str, Any]:
=======
    async def _analyze_primary_complexity(self, query: str, context: Optional[List[str]]) -> Dict[str, Any]:
<<<<<<< HEAD
>>>>>>> 1027e1d0 (Fix linting issues)
        """Analyze primary complexity dimensions."""
=======
        """
        Generate a validated primary complexity assessment for a given research query.
        
        Builds an enhanced prompt from the query and optional context, calls the configured LLM to evaluate ten complexity dimensions and related factors, parses the JSON response, and normalizes the results via the internal validator to produce a combined complexity + recommendation dictionary.
        
        Parameters:
            query: The research question or topic to analyze.
            context: Optional list of context snippets to include in the prompt (used to create a Context Preview).
        
        Returns:
            A dict containing normalized complexity metrics, confidence, adaptive research recommendations, and metadata (timestamp/version). Returns an empty dict if the LLM call or parsing/validation fails (errors are logged rather than raised).
        """
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        try:
<<<<<<< HEAD
            from ..utils.llm import create_chat_completion

<<<<<<< HEAD
=======
            from gpt_researcher.utils.llm import create_chat_completion
            
>>>>>>> 1f1be9a8 (updating)
=======
>>>>>>> 1027e1d0 (Fix linting issues)
            analysis_prompt = self._create_enhanced_complexity_prompt(query, context)

            response = await create_chat_completion(
                model=self.config.smart_llm_model,
                messages=[
                    {
                        "role": "system",
<<<<<<< HEAD
                        "content": "You are an expert research analyst specializing in topic complexity assessment and research planning.",
=======
                        "content": "You are an expert research analyst specializing in topic complexity assessment and research planning."
>>>>>>> 1027e1d0 (Fix linting issues)
                    },
                    {"role": "user", "content": analysis_prompt},
                ],
                temperature=0.2,  # Lower temperature for more consistent analysis
                llm_provider=self.config.smart_llm_provider,
                max_tokens=1500,
                llm_kwargs=self.config.llm_kwargs,
            )

            # Parse and validate response
            complexity_data = json.loads(response)
            return self._validate_and_enhance_complexity_data(complexity_data)

        except Exception as e:
            logger.error(f"Primary complexity analysis failed: {e}")
            return {}

<<<<<<< HEAD
    async def _analyze_domain_specifics(self, query: str) -> dict[str, Any]:
=======
    async def _analyze_domain_specifics(self, query: str) -> Dict[str, Any]:
<<<<<<< HEAD
>>>>>>> 1027e1d0 (Fix linting issues)
        """Analyze domain-specific complexity factors."""
=======
        """
        Analyze domain-specific complexity factors for a research topic and return a structured domain analysis.
        
        Performs an LLM-based classification of the topic's domain characteristics and returns a JSON-like dict with keys:
        - domain_category (str): broad category such as "scientific", "legal", "business", "social", "technical", etc.
        - expertise_required (str): estimated expertise level ("beginner", "intermediate", "expert", "specialist").
        - domain_challenges (List[str]): typical research challenges in the domain.
        - data_availability (str): one of "high", "medium", or "low".
        - methodological_considerations (List[str]): common methodological approaches or constraints.
        
        Parameters:
            query (str): The research topic or question to classify. Used verbatim in the prompt sent to the LLM.
        
        Returns:
            Dict[str, Any]: Parsed domain analysis matching the structure above. If the LLM call fails or returns invalid data, returns a safe empty domain analysis skeleton produced by _get_empty_domain_analysis().
        """
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        try:
<<<<<<< HEAD
            from ..utils.llm import create_chat_completion

<<<<<<< HEAD
=======
            from gpt_researcher.utils.llm import create_chat_completion
            
>>>>>>> 1f1be9a8 (updating)
=======
>>>>>>> 1027e1d0 (Fix linting issues)
            domain_prompt = f"""
Identify domain-specific complexity factors for this research topic:

Topic: {query}

Analyze:
1. Domain category (scientific, legal, business, social, technical, etc.)
2. Required expertise level (beginner, intermediate, expert, specialist)
3. Typical research challenges in this domain
4. Data/source availability in this domain
5. Common methodological approaches

Respond in JSON format:
{{
    "domain_category": "category",
    "expertise_required": "level",
    "domain_challenges": ["challenge1", "challenge2"],
    "data_availability": "high/medium/low",
    "methodological_considerations": ["consideration1", "consideration2"]
}}
"""

            response = await create_chat_completion(
                model=self.config.fast_llm_model,  # Use faster model for domain analysis
                messages=[
                    {
                        "role": "system",
                        "content": "You are a domain classification expert.",
                    },
                    {"role": "user", "content": domain_prompt},
                ],
                temperature=0.3,
                llm_provider=self.config.fast_llm_provider,
                max_tokens=500,
                llm_kwargs=self.config.llm_kwargs,
            )

            return json.loads(response)

        except Exception as e:
            logger.error(f"Domain analysis failed: {e}")
            return self._get_empty_domain_analysis()

<<<<<<< HEAD
    async def _analyze_research_feasibility(
        self, query: str, context: list[str] | None
    ) -> dict[str, Any]:
=======
    async def _analyze_research_feasibility(self, query: str, context: Optional[List[str]]) -> Dict[str, Any]:
<<<<<<< HEAD
>>>>>>> 1027e1d0 (Fix linting issues)
        """Analyze research feasibility and resource requirements."""
=======
        """
        Estimate research feasibility and approximate resource needs for a query.
        
        Uses lightweight heuristics over the query text (length and presence of cue words) to infer scope, depth, and time requirements and to flag whether comparisons or historical analysis are likely required.
        
        Returns:
            A dict with these keys:
            - scope_score (float): 0–10 estimate of scope/complexity (higher = broader/more complex).
            - research_depth (str): 'deep' or 'standard' recommendation for research depth.
            - time_requirement (str): 'extended' or 'normal' estimate of time needed.
            - requires_comparison (bool): True if the query likely asks for comparative analysis.
            - requires_historical (bool): True if the query likely requires historical/temporal analysis.
            - estimated_sources_needed (int): Rough count of sources suggested for adequate coverage.
        
        The function handles internal errors and returns an empty dict on failure.
        """
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        try:
            # Estimate based on query characteristics
            query_length = len(query.split())
<<<<<<< HEAD
            has_specific_terms = any(
                term in query.lower()
                for term in [
                    "specific",
                    "detailed",
                    "comprehensive",
                    "in-depth",
                    "thorough",
                ]
            )
            has_comparative = any(
                term in query.lower()
                for term in ["compare", "versus", "vs", "difference", "contrast"]
            )
            has_temporal = any(
                term in query.lower()
                for term in [
                    "history",
                    "evolution",
                    "timeline",
                    "development",
                    "future",
                ]
            )

            # Calculate feasibility scores
            scope_score = min(10, query_length * 0.5 + (5 if has_specific_terms else 0))
            research_depth = "deep" if has_specific_terms else "standard"
            time_requirement = "extended" if has_temporal else "normal"
=======
            has_specific_terms = any(term in query.lower() for term in [
                'specific', 'detailed', 'comprehensive', 'in-depth', 'thorough'
            ])
            has_comparative = any(term in query.lower() for term in [
                'compare', 'versus', 'vs', 'difference', 'contrast'
            ])
            has_temporal = any(term in query.lower() for term in [
                'history', 'evolution', 'timeline', 'development', 'future'
            ])

            # Calculate feasibility scores
            scope_score = min(10, query_length * 0.5 + (5 if has_specific_terms else 0))
            research_depth = 'deep' if has_specific_terms else 'standard'
            time_requirement = 'extended' if has_temporal else 'normal'
>>>>>>> 1027e1d0 (Fix linting issues)

            return {
                "scope_score": scope_score,
                "research_depth": research_depth,
                "time_requirement": time_requirement,
                "requires_comparison": has_comparative,
                "requires_historical": has_temporal,
                "estimated_sources_needed": 10 + int(scope_score * 2),
            }

        except Exception as e:
            logger.error(f"Feasibility analysis failed: {e}")
            return {}

<<<<<<< HEAD
    def _create_enhanced_complexity_prompt(
        self, query: str, context: list[str] | None
    ) -> str:
        """Create enhanced prompt for complexity analysis."""
        context_text = (
            "\n".join(context[:5]) if context else "No additional context provided."
        )
=======
    def _create_enhanced_complexity_prompt(self, query: str, context: Optional[List[str]]) -> str:
        """
        Builds a deterministic, LLM-ready prompt asking the model to evaluate the research query across ten complexity dimensions and produce a strict JSON response.
        
        The returned prompt includes:
        - The topic and a "Context Preview" made from at most the first five context entries (or a placeholder if none).
        - Definitions for ten scored dimensions (0–10) covering controversy, perspectives, legal certainty, novelty, technical complexity, scope, data availability, interdisciplinarity, temporal relevance, and regulatory complexity.
        - Instructions to produce adaptive recommendations (word-count range, suggested iterations, special attention areas, recommended methodology, and fact-checking rigor).
        - A required exact JSON output schema with numeric ranges for scores, an overall_complexity (0–10), confidence_score (0–1), recommendation fields, and a textual rationale.
        
        Note: the prompt enforces a strict JSON format to simplify parsing by downstream logic and uses the provided context truncated to five items for brevity.
        """
        context_text = "\n".join(context[:5]) if context else "No additional context provided."
>>>>>>> 1027e1d0 (Fix linting issues)

        return f"""
Analyze the complexity of this research topic comprehensively:

Topic: {query}

Context Preview: {context_text}

Evaluate these enhanced dimensions (0-10 scale):

CORE COMPLEXITY FACTORS:
1. Controversy level: Degree of debate, disagreement, or conflicting viewpoints
2. Multiple perspectives: Number of distinct stakeholder views or theoretical frameworks
3. Legal/Regulatory certainty: Clarity and stability of applicable laws/regulations (10 = very clear)
4. Novelty: How emerging or cutting-edge the topic is
5. Technical complexity: Level of specialized knowledge required

ADDITIONAL FACTORS:
6. Scope breadth: How broad vs. narrow the topic is
7. Data availability: Ease of finding reliable sources (10 = abundant sources)
8. Interdisciplinary nature: Degree of cross-field knowledge required
9. Temporal relevance: Time-sensitivity or historical span required
10. Regulatory complexity: Intricacy of compliance/regulatory aspects

RECOMMENDATIONS:
Based on your analysis, provide:
- Word count range (considering complexity)
- Number of research iterations needed
- Critical areas requiring special attention
- Suggested research methodology
- Fact-checking rigor level needed

Respond in this exact JSON format:
{{
    "controversy_level": 0-10,
    "multiple_perspectives": 0-10,
    "legal_certainty": 0-10,
    "novelty": 0-10,
    "technical_complexity": 0-10,
    "scope_breadth": 0-10,
    "data_availability": 0-10,
    "interdisciplinary_nature": 0-10,
    "temporal_relevance": 0-10,
    "regulatory_complexity": 0-10,
    "overall_complexity": 0-10,
    "confidence_score": 0-1,
    "recommended_min_words": number,
    "recommended_max_words": number,
    "suggested_iterations": number,
    "special_attention_areas": ["area1", "area2", "area3"],
    "recommended_methodology": "qualitative/quantitative/mixed",
    "fact_checking_rigor": "standard/enhanced/maximum",
    "rationale": "Detailed explanation of the complexity assessment and recommendations"
}}
"""

<<<<<<< HEAD
    def _validate_and_enhance_complexity_data(
        self, data: dict[str, Any]
    ) -> dict[str, Any]:
=======
    def _validate_and_enhance_complexity_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
<<<<<<< HEAD
>>>>>>> 1027e1d0 (Fix linting issues)
        """Validate, normalize, and enhance complexity data."""
=======
        """
        Validate, normalize, and enrich raw complexity output into a canonical analysis result.
        
        Takes a raw dictionary `data` (typically produced by an LLM) and returns a deterministic, validated analysis
        containing normalized per-dimension metrics, adaptive recommendations, and metadata suitable for downstream
        consumption.
        
        Parameters:
            data (Dict[str, Any]): Raw complexity fields (numeric scores and optional recommendation overrides).
                Numeric metric values will be clamped to the 0–10 range. Known metric keys correspond to the
                ComplexityMetrics dataclass fields (ten dimension scores, `overall_complexity`, `confidence_score`, etc.).
                Recommendation-related overrides present in `data` (e.g., word count or iterations) will be respected
                when building the adaptive recommendations.
        
        Returns:
            Dict[str, Any]: A merged dictionary containing:
              - Normalized complexity metrics (all ComplexityMetrics fields).
              - Generated ResearchRecommendations fields.
              - analysis_timestamp (ISO 8601 string) and analysis_version (string).
        
        Behavior notes (concise):
          - Any numeric metric provided is clamped to [0, 10].
          - If `overall_complexity` is missing or equal to the default sentinel (5), an overall score is computed
            from the other metrics using the analyzer's weighting logic.
          - `confidence_score` is set to the fraction of known metric fields present in `data`.
          - Recommendations are produced by the analyzer's adaptive rule set and may incorporate overrides from `data`.
        """
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        # Create metrics object
        metrics = ComplexityMetrics()

        # Update metrics with provided data
        for field in metrics.__dataclass_fields__:
            if field in data:
                value = data[field]
                if isinstance(value, int | float):
                    # Normalize to 0-10 range
                    normalized = max(0, min(10, value))
                    setattr(metrics, field, normalized)

        # Calculate overall complexity if not provided
        if "overall_complexity" not in data or data["overall_complexity"] == 5:
            metrics.overall_complexity = self._calculate_overall_complexity(metrics)

        # Calculate confidence score based on data completeness
        provided_fields = sum(1 for f in metrics.__dataclass_fields__ if f in data)
        total_fields = len(metrics.__dataclass_fields__)
        metrics.confidence_score = provided_fields / total_fields

        # Create recommendations based on complexity
        recommendations = self._generate_recommendations(metrics, data)

        # Combine metrics and recommendations
        result = asdict(metrics)
        result.update(asdict(recommendations))

        # Add additional metadata
<<<<<<< HEAD
        result["analysis_timestamp"] = datetime.now().isoformat()
        result["analysis_version"] = "2.0"
=======
        result['analysis_timestamp'] = datetime.now().isoformat()
        result['analysis_version'] = '2.0'
>>>>>>> 1027e1d0 (Fix linting issues)

        return result

    def _calculate_overall_complexity(self, metrics: ComplexityMetrics) -> float:
        """
        Compute a single weighted overall complexity score from per-dimension metrics.
        
        The function applies predefined weights to selected complexity dimensions in the provided
        ComplexityMetrics and returns a final score on a 0–10 scale. The `data_availability`
        dimension is inverted (higher availability reduces complexity) before weighting. The
        result is rounded to one decimal place.
        
        Parameters:
            metrics (ComplexityMetrics): Per-dimension scores (0–10) used to compute the aggregate.
        
        Returns:
            float: Weighted overall complexity score (0.0–10.0), rounded to one decimal place.
        """
        weights = {
            "controversy_level": 0.15,
            "multiple_perspectives": 0.15,
            "technical_complexity": 0.20,
            "scope_breadth": 0.10,
            "data_availability": 0.10,  # Inverted - low availability = high complexity
            "interdisciplinary_nature": 0.10,
            "novelty": 0.10,
            "regulatory_complexity": 0.10,
        }

        weighted_sum = 0
        for field, weight in weights.items():
            value = getattr(metrics, field, 5)
            # Invert data_availability (high availability = low complexity)
            if field == "data_availability":
                value = 10 - value
            weighted_sum += value * weight

        return round(weighted_sum, 1)

<<<<<<< HEAD
    def _generate_recommendations(
        self, metrics: ComplexityMetrics, raw_data: dict
    ) -> ResearchRecommendations:
=======
    def _generate_recommendations(self, metrics: ComplexityMetrics, raw_data: Dict) -> ResearchRecommendations:
<<<<<<< HEAD
>>>>>>> 1027e1d0 (Fix linting issues)
        """Generate adaptive recommendations based on complexity metrics."""
=======
        """
        Create adaptive, bounded research recommendations from computed complexity metrics.
        
        Produces a ResearchRecommendations object whose values are computed from the provided
        ComplexityMetrics and may be overridden or adjusted by entries in raw_data.
        
        Detailed behavior:
        - Word count: selects a recommended min/max range based on overall complexity, with clamped overrides from raw_data:
          - Defaults scale from lightweight (800–1500) up to very in-depth (4000–8000).
          - Overrides are clamped to sensible bounds (min >= 500, max >= min+500, max caps applied).
        - Iterations: derives a suggested iteration count from key metrics (controversy, perspectives, scope breadth, technical complexity),
          then applies a cap of 5. A raw_data 'suggested_iterations' value, if present, replaces the computed value (still constrained).
        - Special attention areas: uses raw_data['special_attention_areas'] if provided and non-empty; otherwise auto-generates via _identify_attention_areas(metrics).
        - Report type and depth: maps high technical complexity -> 'technical_report' / 'deep'; high controversy -> 'analytical_report' / 'comprehensive'; otherwise 'standard_report' / 'standard'.
        - Source diversity: set to 'maximum' / 'high' / 'standard' according to multiple_perspectives thresholds.
        - Fact-checking rigor: defaults to raw_data['fact_checking_rigor'] when present; escalates to 'maximum' for high controversy or regulatory complexity, or to 'enhanced' for high overall complexity.
        - Rationale: uses raw_data['rationale'] if provided; otherwise generates one via _generate_rationale(metrics).
        
        Parameters:
        - metrics (ComplexityMetrics): normalized per-dimension scores and computed overall_complexity used to drive recommendations.
        - raw_data (dict): optional values from upstream analysis (possible keys include 'recommended_min_words', 'recommended_max_words',
          'suggested_iterations', 'special_attention_areas', 'fact_checking_rigor', 'rationale') that may override defaults; numeric overrides are clamped for safety.
        
        Returns:
        - ResearchRecommendations: fully populated recommendation object with word ranges, iterations, attention areas, report type/depth,
          source diversity requirement, fact-checking rigor, and a textual rationale.
        """
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        overall = metrics.overall_complexity

        # Adaptive word count based on complexity
        if overall < 3:
            min_words, max_words = 800, 1500
        elif overall < 5:
            min_words, max_words = 1500, 3000
        elif overall < 7:
            min_words, max_words = 2500, 5000
        else:
            min_words, max_words = 4000, 8000

        # Override with provided values if valid
<<<<<<< HEAD
        if "recommended_min_words" in raw_data:
            min_words = max(500, min(10000, raw_data["recommended_min_words"]))
        if "recommended_max_words" in raw_data:
            max_words = max(
                min_words + 500, min(15000, raw_data["recommended_max_words"])
            )
=======
        if 'recommended_min_words' in raw_data:
            min_words = max(500, min(10000, raw_data['recommended_min_words']))
        if 'recommended_max_words' in raw_data:
            max_words = max(min_words + 500, min(15000, raw_data['recommended_max_words']))
>>>>>>> 1027e1d0 (Fix linting issues)

        # Adaptive iterations based on complexity factors
        iterations = 1
        if metrics.controversy_level > 7 or metrics.multiple_perspectives > 7:
            iterations += 1
        if metrics.scope_breadth > 7:
            iterations += 1
        if metrics.technical_complexity > 8:
            iterations += 1
<<<<<<< HEAD
        iterations = min(5, max(1, raw_data.get("suggested_iterations", iterations)))
=======
        iterations = min(5, max(1, raw_data.get('suggested_iterations', iterations)))
>>>>>>> 1027e1d0 (Fix linting issues)

        # Special attention areas
        attention_areas = raw_data.get("special_attention_areas", [])
        if not attention_areas:
            attention_areas = self._identify_attention_areas(metrics)

        # Determine report type and depth
        if metrics.technical_complexity > 7:
            report_type = "technical_report"
            research_depth = "deep"
        elif metrics.controversy_level > 7:
            report_type = "analytical_report"
            research_depth = "comprehensive"
        else:
<<<<<<< HEAD
            report_type = "standard_report"
            research_depth = "standard"
=======
            report_type = 'standard_report'
            research_depth = 'standard'
>>>>>>> 1027e1d0 (Fix linting issues)

        # Source diversity requirements
        if metrics.multiple_perspectives > 7:
            source_diversity = "maximum"
        elif metrics.multiple_perspectives > 5:
            source_diversity = "high"
        else:
<<<<<<< HEAD
            source_diversity = "standard"
=======
            source_diversity = 'standard'
>>>>>>> 1027e1d0 (Fix linting issues)

        # Fact-checking rigor
        fact_checking = raw_data.get("fact_checking_rigor", "standard")
        if metrics.controversy_level > 7 or metrics.regulatory_complexity > 7:
            fact_checking = "maximum"
        elif overall > 6:
<<<<<<< HEAD
            fact_checking = "enhanced"
=======
            fact_checking = 'enhanced'
>>>>>>> 1027e1d0 (Fix linting issues)

        return ResearchRecommendations(
            recommended_min_words=min_words,
            recommended_max_words=max_words,
            suggested_iterations=iterations,
            special_attention_areas=attention_areas,
            recommended_report_type=report_type,
            research_depth=research_depth,
            source_diversity_needed=source_diversity,
            fact_checking_rigor=fact_checking,
            rationale=raw_data.get("rationale", self._generate_rationale(metrics)),
        )

<<<<<<< HEAD
    def _identify_attention_areas(self, metrics: ComplexityMetrics) -> list[str]:
=======
    def _identify_attention_areas(self, metrics: ComplexityMetrics) -> List[str]:
<<<<<<< HEAD
>>>>>>> 1027e1d0 (Fix linting issues)
        """Identify areas needing special attention based on metrics."""
=======
        """
        Return a prioritized list of up to five research or review focus areas that require special attention based on complexity metrics.
        
        Evaluates key ComplexityMetrics fields against thresholds and maps notable signals to human-readable attention areas. Current heuristics:
        - controversy_level > 7 → "Balanced representation of conflicting viewpoints"
        - technical_complexity > 7 → "Technical accuracy and expert validation"
        - regulatory_complexity > 7 → "Regulatory compliance and legal accuracy"
        - multiple_perspectives > 7 → "Stakeholder perspective mapping"
        - data_availability < 3 → "Limited source availability - require thorough search"
        - novelty > 7 → "Emerging topic - verify cutting-edge information"
        - temporal_relevance > 7 → "Historical context and timeline accuracy"
        
        The returned list preserves the detection order and is truncated to at most five entries.
        Returns:
            List[str]: Ordered attention-area descriptions (0–5 items) derived from `metrics`.
        """
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        areas = []

        if metrics.controversy_level > 7:
            areas.append("Balanced representation of conflicting viewpoints")
        if metrics.technical_complexity > 7:
            areas.append("Technical accuracy and expert validation")
        if metrics.regulatory_complexity > 7:
            areas.append("Regulatory compliance and legal accuracy")
        if metrics.multiple_perspectives > 7:
            areas.append("Stakeholder perspective mapping")
        if metrics.data_availability < 3:
            areas.append("Limited source availability - require thorough search")
        if metrics.novelty > 7:
            areas.append("Emerging topic - verify cutting-edge information")
        if metrics.temporal_relevance > 7:
            areas.append("Historical context and timeline accuracy")

        return areas[:5]  # Limit to top 5 areas

    def _generate_rationale(self, metrics: ComplexityMetrics) -> str:
        """
        Compose a concise, human-readable rationale for a complexity assessment.
        
        Given a ComplexityMetrics instance, returns a one-sentence explanation that:
        - Summarizes the overall complexity level (low, moderate, substantial, or high) derived from metrics.overall_complexity.
        - Recommends the general research approach (straightforward, balanced, comprehensive, or intensive) aligned with that level.
        - Appends notable contributing factors when present (high controversy, technical depth, broad scope) if their scores exceed 7.
        
        Parameters:
            metrics (ComplexityMetrics): Per-dimension scores and computed overall_complexity used to derive the rationale.
        
        Returns:
            str: A single-sentence rationale describing the complexity level, recommended approach, and any highlighted factors.
        """
        overall = metrics.overall_complexity

        if overall < 3:
            level = "low complexity"
            approach = "straightforward research approach"
        elif overall < 5:
            level = "moderate complexity"
            approach = "balanced research approach"
        elif overall < 7:
            level = "substantial complexity"
            approach = "comprehensive research approach"
        else:
            level = "high complexity"
            approach = "intensive research approach"

        factors = []
        if metrics.controversy_level > 7:
            factors.append("high controversy")
        if metrics.technical_complexity > 7:
            factors.append("technical depth")
        if metrics.scope_breadth > 7:
            factors.append("broad scope")

        factor_text = f" due to {', '.join(factors)}" if factors else ""

        return f"This topic exhibits {level}{factor_text}, requiring a {approach} with careful attention to accuracy and completeness."

    def _combine_analyses(
<<<<<<< HEAD
        self, primary: dict, domain: dict, feasibility: dict, query: str
    ) -> dict[str, Any]:
=======
        self,
        primary: Dict,
        domain: Dict,
        feasibility: Dict,
        query: str
    ) -> Dict[str, Any]:
<<<<<<< HEAD
>>>>>>> 1027e1d0 (Fix linting issues)
        """Combine multiple analyses into comprehensive result."""
=======
        """
        Merge primary, domain, and feasibility analyses into a single result dictionary.
        
        If `primary` is empty or falsy, a default complexity result is used as the base. When provided, `domain` is attached under the `domain_analysis` key and `feasibility` under `feasibility_analysis`. The feasibility analysis can influence top-level recommendation fields:
        - If `feasibility['requires_comparison']` is truthy, `suggested_iterations` is raised to at least 2.
        - If `feasibility['requires_historical']` is truthy, `recommended_max_words` is raised to at least 3000.
        
        This function also adds query metadata:
        - `query`: the original query string.
        - `query_word_count`: word count computed via splitting on whitespace.
        
        Returns:
            A merged dictionary containing the combined analyses, any adjusted recommendation fields, and query metadata.
        """
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        combined = primary.copy() if primary else self._get_default_complexity()

        # Enhance with domain analysis
        if domain:
<<<<<<< HEAD
            combined["domain_analysis"] = domain
=======
            combined['domain_analysis'] = domain
>>>>>>> 1027e1d0 (Fix linting issues)

        # Enhance with feasibility analysis
        if feasibility:
            combined["feasibility_analysis"] = feasibility
            # Adjust recommendations based on feasibility
<<<<<<< HEAD
            if feasibility.get("requires_comparison"):
                combined["suggested_iterations"] = max(
                    2, combined.get("suggested_iterations", 1)
                )
            if feasibility.get("requires_historical"):
                combined["recommended_max_words"] = max(
                    3000, combined.get("recommended_max_words", 2000)
                )

        # Add query metadata
        combined["query"] = query
        combined["query_word_count"] = len(query.split())
=======
            if feasibility.get('requires_comparison'):
                combined['suggested_iterations'] = max(2, combined.get('suggested_iterations', 1))
            if feasibility.get('requires_historical'):
                combined['recommended_max_words'] = max(3000, combined.get('recommended_max_words', 2000))

        # Add query metadata
        combined['query'] = query
        combined['query_word_count'] = len(query.split())
>>>>>>> 1027e1d0 (Fix linting issues)

        return combined

    def _is_complexity_analysis_enabled(self) -> bool:
        """Check if complexity analysis is enabled in configuration."""
<<<<<<< HEAD
        complexity_factors = getattr(self.config, "complexity_factors", {})
        return complexity_factors.get(
            "controversy_detection", False
        ) or complexity_factors.get("enable_complexity_analysis", False)

    def _generate_cache_key(self, query: str, context: list[str] | None) -> str:
=======
        complexity_factors = getattr(self.config, 'complexity_factors', {})
        return complexity_factors.get('controversy_detection', False) or \
               complexity_factors.get('enable_complexity_analysis', False)

    def _generate_cache_key(self, query: str, context: Optional[List[str]]) -> str:
<<<<<<< HEAD
>>>>>>> 1027e1d0 (Fix linting issues)
        """Generate cache key for complexity analysis."""
=======
        """
        Create a deterministic cache key for a query and optional context.
        
        The key is the query string followed by a compact representation of up to the first three
        context items (if provided), joined with a colon. This ensures similar queries with the
        same leading context map to the same cache entry while keeping keys concise.
        """
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        context_str = str(context[:3]) if context else ""
        return f"{query}:{context_str}"

<<<<<<< HEAD
    def _get_empty_domain_analysis(self) -> dict[str, Any]:
=======
    def _get_empty_domain_analysis(self) -> Dict[str, Any]:
<<<<<<< HEAD
>>>>>>> 1027e1d0 (Fix linting issues)
        """Return empty domain analysis structure."""
=======
        """
        Return a default domain analysis skeleton used when domain-specific inference is unavailable or an error occurs.
        
        Returns a dict with the following keys:
        - domain_category (str): high-level domain label (default "general").
        - expertise_required (str): suggested expertise level (default "intermediate").
        - domain_challenges (List[str]): notable domain challenges (empty list by default).
        - data_availability (str): qualitative estimate of data availability (default "medium").
        - methodological_considerations (List[str]): recommended methodological notes (empty list by default).
        """
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        return {
            "domain_category": "general",
            "expertise_required": "intermediate",
            "domain_challenges": [],
            "data_availability": "medium",
            "methodological_considerations": [],
        }

<<<<<<< HEAD
    def _get_default_complexity(self) -> dict[str, Any]:
=======
    def _get_default_complexity(self) -> Dict[str, Any]:
<<<<<<< HEAD
>>>>>>> 1027e1d0 (Fix linting issues)
        """Return enhanced default complexity analysis."""
=======
        """
        Return a default enhanced complexity analysis result used when analysis is disabled or fails.
        
        This result contains:
        - Default per-dimension complexity scores (from ComplexityMetrics defaults).
        - Default adaptive recommendations (from ResearchRecommendations), where
          recommended_min_words is taken from self.config.total_words if present, otherwise 2000,
          and recommended_max_words is twice that value.
        - Metadata fields: analysis_timestamp (ISO 8601), analysis_version ('2.0'), and from_cache (False).
        
        Returns:
            Dict[str, Any]: A merged dictionary of metrics, recommendations, and metadata representing a safe default complexity analysis.
        """
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        metrics = ComplexityMetrics()
        recommendations = ResearchRecommendations(
            recommended_min_words=getattr(self.config, "total_words", 2000),
            recommended_max_words=getattr(self.config, "total_words", 2000) * 2,
            suggested_iterations=1,
            special_attention_areas=[],
            recommended_report_type="standard_report",
            research_depth="standard",
            source_diversity_needed="standard",
            fact_checking_rigor="standard",
            rationale="Default complexity assessment - analysis disabled or failed",
        )

        result = asdict(metrics)
        result.update(asdict(recommendations))
<<<<<<< HEAD
        result["analysis_timestamp"] = datetime.now().isoformat()
        result["analysis_version"] = "2.0"
        result["from_cache"] = False

        return result

    def get_complexity_history(self) -> list[dict]:
=======
        result['analysis_timestamp'] = datetime.now().isoformat()
        result['analysis_version'] = '2.0'
        result['from_cache'] = False

        return result

    def get_complexity_history(self) -> List[Dict]:
<<<<<<< HEAD
>>>>>>> 1027e1d0 (Fix linting issues)
        """Get history of complexity analyses performed."""
=======
        """
        Return the recorded history of complexity analyses.
        
        Each entry is a dict recorded when an analysis completed (chronological order). Entries typically include keys such as `analysis_timestamp`, `overall_complexity`, and any metadata produced by the analyzer. The returned list is the internal history structure (mutable reference).
        """
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        return self.analysis_history

    def clear_cache(self):
        """
        Clear the in-memory analysis cache.
        
        This removes all cached complexity analysis results stored on the instance. Intended to free memory or force fresh recomputation on subsequent calls to analyze_complexity. The method also emits an informational log entry when the cache is cleared.
        """
        self.cache.clear()
        logger.info("Complexity analysis cache cleared")
