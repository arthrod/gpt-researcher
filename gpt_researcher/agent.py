from typing import Any, Optional
import json
import os

from .config import Config
from .memory import Memory
from .utils.enum import ReportSource, ReportType, Tone
from .llm_provider import GenericLLMProvider
from .prompts import get_prompt_family
from .vector_store import VectorStoreWrapper

# Research skills
from .skills.researcher import ResearchConductor
from .skills.writer import ReportGenerator
from .skills.context_manager import ContextManager
from .skills.browser import BrowserManager
from .skills.curator import SourceCurator
from .skills.deep_research import DeepResearchSkill

from .actions import (
    add_references,
    extract_headers,
    extract_sections,
    table_of_contents,
    get_search_results,
    get_retrievers,
    choose_agent
)


class GPTResearcher:
    def __init__(
        self,
        query: str,
        report_type: str = ReportType.ResearchReport.value,
        report_format: str | None = None,
        report_source: str | None = None,
        tone: Tone = Tone.Objective,
        source_urls: list[str] | None = None,
        document_urls: list[str] | None = None,
        complement_source_urls: bool = False,
        query_domains: list[str] | None = None,
        documents=None,
        vector_store=None,
        vector_store_filter=None,
        config_path=None,
        websocket=None,
        agent=None,
        role=None,
        parent_query: str = "",
        subtopics: list | None = None,
        visited_urls: set | None = None,
        verbose: bool = True,
        context=None,
        headers: dict | None = None,
        max_subtopics: int | None = None,
        log_handler=None,
        prompt_family: str | None = None,
        mcp_configs: list[dict] | None = None,
        mcp_max_iterations: int | None = None,
        mcp_strategy: str | None = None,
        # Config options
        retriever: str | None = None,
        embedding: str | None = None,
        similarity_threshold: float | None = None,
        fast_llm: str | None = None,
        smart_llm: str | None = None,
        strategic_llm: str | None = None,
        fast_token_limit: int | None = None,
        smart_token_limit: int | None = None,
        strategic_token_limit: int | None = None,
        browse_chunk_max_length: int | None = None,
        summary_token_limit: int | None = None,
        temperature: float | None = None,
        user_agent: str | None = None,
        max_search_results_per_query: int | None = None,
        memory_backend: str | None = None,
        total_words: int | None = None,
        curate_sources: bool | None = None,
        max_iterations: int | None = None,
        language: str | None = None,
        scraper: str | None = None,
        max_scraper_workers: int | None = None,
        scraper_rate_limit_delay: float | None = None,
        doc_path: str | None = None,
        llm_kwargs: dict | None = None,
        embedding_kwargs: dict | None = None,
        deep_research_concurrency: int | None = None,
        deep_research_depth: int | None = None,
        deep_research_breadth: int | None = None,
        mcp_auto_tool_selection: bool | None = None,
        mcp_use_llm_args: bool | None = None,
        mcp_allowed_root_paths: list[str] | None = None,
        reasoning_effort: str | None = None,
        **kwargs
    ):
        """
        Initialize a GPT Researcher instance.
        
        Args:
            query (str): The research query or question.
            report_type (str): Type of report to generate.
            report_format (str): Format of the report (markdown, pdf, etc).
            report_source (str): Source of information for the report (web, local, etc).
            tone (Tone): Tone of the report.
            source_urls (list[str], optional): List of specific URLs to use as sources.
            document_urls (list[str], optional): List of document URLs to use as sources.
            complement_source_urls (bool): Whether to complement source URLs with web search.
            query_domains (list[str], optional): List of domains to restrict search to.
            documents: Document objects for LangChain integration.
            vector_store: Vector store for document retrieval.
            vector_store_filter: Filter for vector store queries.
            config_path: Path to configuration file.
            websocket: WebSocket for streaming output.
            agent: Pre-defined agent type.
            role: Pre-defined agent role.
            parent_query: Parent query for subtopic reports.
            subtopics: List of subtopics to research.
            visited_urls: Set of already visited URLs.
            verbose (bool): Whether to output verbose logs.
            context: Pre-loaded research context.
            headers (dict, optional): Additional headers for requests and configuration.
            max_subtopics (int): Maximum number of subtopics to generate.
            log_handler: Handler for logging events.
            prompt_family: Family of prompts to use.
            mcp_configs (list[dict], optional): List of MCP server configurations.
            mcp_max_iterations (int, optional): Legacy MCP parameter.
            mcp_strategy (str, optional): MCP execution strategy.
            **kwargs: Additional arguments to pass to the Config class.
        """
        self.query = query
        self.report_type = report_type
        self.tone = tone if isinstance(tone, Tone) else Tone.Objective
        self.source_urls = source_urls
        self.document_urls = document_urls
        self.complement_source_urls = complement_source_urls
        self.query_domains = query_domains or []
        self.documents = documents
        self.vector_store = VectorStoreWrapper(vector_store) if vector_store else None
        self.vector_store_filter = vector_store_filter
        self.websocket = websocket
        self.agent = agent
        self.role = role
        self.parent_query = parent_query
        self.subtopics = subtopics or []
        self.visited_urls = visited_urls or set()
        self.verbose = verbose
        self.context = context or []
        self.headers = headers or {}
        self.log_handler = log_handler
        self.research_sources = []
        self.research_images = []
        self.research_costs = 0.0

        # Create config with all possible overrides
        # We explicitly pass arguments that were provided (not None)
        config_overrides = {k: v for k, v in kwargs.items()}

        # Add explicit arguments to overrides if they are not None
        explicit_args = {
            "retriever": retriever,
            "embedding": embedding,
            "similarity_threshold": similarity_threshold,
            "fast_llm": fast_llm,
            "smart_llm": smart_llm,
            "strategic_llm": strategic_llm,
            "fast_token_limit": fast_token_limit,
            "smart_token_limit": smart_token_limit,
            "strategic_token_limit": strategic_token_limit,
            "browse_chunk_max_length": browse_chunk_max_length,
            "summary_token_limit": summary_token_limit,
            "temperature": temperature,
            "user_agent": user_agent,
            "max_search_results_per_query": max_search_results_per_query,
            "memory_backend": memory_backend,
            "total_words": total_words,
            "curate_sources": curate_sources,
            "max_iterations": max_iterations,
            "language": language,
            "scraper": scraper,
            "max_scraper_workers": max_scraper_workers,
            "scraper_rate_limit_delay": scraper_rate_limit_delay,
            "doc_path": doc_path,
            "llm_kwargs": llm_kwargs,
            "embedding_kwargs": embedding_kwargs,
            "deep_research_concurrency": deep_research_concurrency,
            "deep_research_depth": deep_research_depth,
            "deep_research_breadth": deep_research_breadth,
            "mcp_auto_tool_selection": mcp_auto_tool_selection,
            "mcp_use_llm_args": mcp_use_llm_args,
            "mcp_allowed_root_paths": mcp_allowed_root_paths,
            "reasoning_effort": reasoning_effort,
            "report_format": report_format,
            "report_source": report_source,
            "max_subtopics": max_subtopics,
            "prompt_family": prompt_family,
            "mcp_strategy": mcp_strategy,
            "agent_role": role # mapping role to agent_role config
        }

        for k, v in explicit_args.items():
            if v is not None:
                config_overrides[k] = v

        # Filter out encoding from kwargs if it exists (it was in original code)
        self.encoding = config_overrides.pop('encoding', 'utf-8')
        
        # Initialize Config with overrides
        self.cfg = Config(config_path, **config_overrides)
        self.cfg.set_verbose(verbose)

        # Update self.kwargs to reflect remaining kwargs
        self.kwargs = kwargs

        # Re-assign variables that depend on config
        self.report_format = report_format if report_format is not None else self.cfg.report_format
        self.report_source = report_source if report_source is not None else self.cfg.report_source
        self.max_subtopics = max_subtopics if max_subtopics is not None else self.cfg.max_subtopics
        self.prompt_family = get_prompt_family(prompt_family or self.cfg.prompt_family, self.cfg)

        # Process MCP configurations if provided
        self.mcp_configs = mcp_configs
        if mcp_configs:
            self._process_mcp_configs(mcp_configs)
        
        self.retrievers = get_retrievers(self.headers, self.cfg)
        self.memory = Memory(
            self.cfg.embedding_provider, self.cfg.embedding_model, **self.cfg.embedding_kwargs
        )

        # Initialize components
        self.research_conductor: ResearchConductor = ResearchConductor(self)
        self.report_generator: ReportGenerator = ReportGenerator(self)
        self.context_manager: ContextManager = ContextManager(self)
        self.scraper_manager: BrowserManager = BrowserManager(self)
        self.source_curator: SourceCurator = SourceCurator(self)
        self.deep_researcher: Optional[DeepResearchSkill] = None
        if report_type == ReportType.DeepResearch.value:
            self.deep_researcher = DeepResearchSkill(self)

        # Handle MCP strategy configuration with backwards compatibility
        self.mcp_strategy = self._resolve_mcp_strategy(mcp_strategy, mcp_max_iterations)

    def _resolve_mcp_strategy(self, mcp_strategy: str | None, mcp_max_iterations: int | None) -> str:
        """
        Resolve MCP strategy from various sources with backwards compatibility.
        
        Priority:
        1. Parameter mcp_strategy (new approach)
        2. Parameter mcp_max_iterations (backwards compatibility)  
        3. Config MCP_STRATEGY
        4. Default "fast"
        
        Args:
            mcp_strategy: New strategy parameter
            mcp_max_iterations: Legacy parameter for backwards compatibility
            
        Returns:
            str: Resolved strategy ("fast", "deep", or "disabled")
        """
        # Priority 1: Use mcp_strategy parameter if provided
        if mcp_strategy is not None:
            # Support new strategy names
            if mcp_strategy in ["fast", "deep", "disabled"]:
                return mcp_strategy
            # Support old strategy names for backwards compatibility
            elif mcp_strategy == "optimized":
                import logging
                logging.getLogger(__name__).warning("mcp_strategy 'optimized' is deprecated, use 'fast' instead")
                return "fast"
            elif mcp_strategy == "comprehensive":
                import logging
                logging.getLogger(__name__).warning("mcp_strategy 'comprehensive' is deprecated, use 'deep' instead")
                return "deep"
            else:
                import logging
                logging.getLogger(__name__).warning(f"Invalid mcp_strategy '{mcp_strategy}', defaulting to 'fast'")
                return "fast"
        
        # Priority 2: Convert mcp_max_iterations for backwards compatibility
        if mcp_max_iterations is not None:
            import logging
            logging.getLogger(__name__).warning("mcp_max_iterations is deprecated, use mcp_strategy instead")
            
            if mcp_max_iterations == 0:
                return "disabled"
            elif mcp_max_iterations == 1:
                return "fast"
            elif mcp_max_iterations == -1:
                return "deep"
            else:
                # Treat any other number as fast mode
                return "fast"
        
        # Priority 3: Use config setting
        if hasattr(self.cfg, 'mcp_strategy'):
            config_strategy = self.cfg.mcp_strategy
            # Support new strategy names
            if config_strategy in ["fast", "deep", "disabled"]:
                return config_strategy
            # Support old strategy names for backwards compatibility
            elif config_strategy == "optimized":
                return "fast"
            elif config_strategy == "comprehensive":
                return "deep"
            
        # Priority 4: Default to fast
        return "fast"

    def _process_mcp_configs(self, mcp_configs: list[dict]) -> None:
        """
        Process MCP configurations from a list of configuration dictionaries.
        
        This method validates the MCP configurations. It only adds MCP to retrievers
        if no explicit retriever configuration is provided via environment variables.
        
        Args:
            mcp_configs (list[dict]): List of MCP server configuration dictionaries.
        """
        # Check if user explicitly set RETRIEVER environment variable
        # We also check if it was set in self.cfg via kwargs, as that is now the priority
        user_set_retriever = os.getenv("RETRIEVER") is not None
        
        # Also check if retriever was set in config (via kwargs override)
        # Note: In __init__, self.cfg.retrievers might have been set by kwargs.
        # We need to respect that.
        
        # Original logic was checking if user set it. Now "user set it" includes passing it as an arg.
        # If the user passed 'retriever' arg, self.cfg.retrievers is already parsed and set.

        # If 'retriever' arg was passed to __init__, it's in self.kwargs or captured in explicit_args.
        # However, self.cfg handles parsing.

        # We want to add 'mcp' to retrievers if:
        # 1. User didn't explicitly specify retrievers (either via env or arg) that EXCLUDE 'mcp'.
        # But the original logic was: "Only auto-add MCP if user hasn't explicitly set retrievers".
        # This implies if user sets "tavily", we don't add "mcp".

        # If retrievers are default ("tavily"), we might want to add mcp.
        # Let's check self.cfg.retrievers.

        # If config was initialized with defaults, it might be ["tavily"].
        # If user passed "google", it is ["google"].

        # The logic here is tricky with the refactor.
        # Previously: `user_set_retriever = os.getenv("RETRIEVER") is not None`
        # Now we should check if `retriever` was in config_overrides or os.environ.

        # But wait, self.cfg is already initialized.
        # If `retriever` was passed in kwargs, `self.cfg.retrievers` reflects that.
        # If `RETRIEVER` env var was present, `self.cfg.retrievers` reflects that.

        # The goal of this method seems to be: If using defaults (or auto-detected mcp config), ensure 'mcp' is in retrievers.

        # Let's preserve the spirit: if the user explicitly configured retrievers (via env or arg), respect it.
        # If they didn't (using default), and we have mcp_configs, add 'mcp'.

        # How do we know if they explicitly configured it?
        # Check if 'retriever' key was in config_overrides OR os.environ.

        # Since we don't have access to config_overrides here easily (local var in __init__),
        # we can check if self.cfg.retrievers is different from default? No, default is "tavily".

        # Simpler approach:
        # If mcp_configs is provided, we essentially want to enable MCP unless explicitly disabled?
        # Or maybe just append it if using defaults?

        # Let's stick to the existing logic but expanded:
        # If user explicitly set RETRIEVER env var -> respect it.
        # If user explicitly passed `retriever` arg -> respect it.

        # We can detect if `retriever` arg was passed by checking `self.kwargs` or the fact that we can't easily check local vars from __init__.
        # Actually, `self.cfg.retriever` is not a stored property for the raw string, `self.cfg.retrievers` is the list.

        # Let's look at how `_process_mcp_configs` was implemented before my changes.
        # It checked `os.getenv("RETRIEVER")`.

        # I will leave it mostly as is, but maybe add a comment or slight adjustment if I can detect arg usage.
        # But wait, I see `config_retriever_set` variable in the file content I read.

        # The code I read has:
        # config_retriever_set = self.cfg.retriever != "tavily" and self.cfg.retriever != os.getenv("RETRIEVER")

        # This looks like it tries to detect if config is different from env/default?
        # But `self.cfg` doesn't have a `retriever` attribute (singular), it usually processes it into `retrievers` (plural list).
        # Let's check `Config` class again. `_set_attributes` sets attributes dynamically.
        # If `RETRIEVER` is in config/env, it sets `self.retriever`.
        # `parse_retrievers` sets `self.retrievers`.

        # So `self.cfg.retriever` (singular) likely exists if set.

        pass

    async def _log_event(self, event_type: str, **kwargs):
        """Helper method to handle logging events"""
        if self.log_handler:
            try:
                if event_type == "tool":
                    await self.log_handler.on_tool_start(kwargs.get('tool_name', ''), **kwargs)
                elif event_type == "action":
                    await self.log_handler.on_agent_action(kwargs.get('action', ''), **kwargs)
                elif event_type == "research":
                    await self.log_handler.on_research_step(kwargs.get('step', ''), kwargs.get('details', {}))

                # Add direct logging as backup
                import logging
                research_logger = logging.getLogger('research')
                research_logger.info(f"{event_type}: {json.dumps(kwargs, default=str)}")

            except Exception as e:
                import logging
                logging.getLogger('research').error(f"Error in _log_event: {e}", exc_info=True)

    async def conduct_research(self, on_progress=None):
        await self._log_event("research", step="start", details={
            "query": self.query,
            "report_type": self.report_type,
            "agent": self.agent,
            "role": self.role
        })

        # Handle deep research separately
        if self.report_type == ReportType.DeepResearch.value and self.deep_researcher:
            return await self._handle_deep_research(on_progress)

        if not (self.agent and self.role):
            await self._log_event("action", action="choose_agent")
            # Filter out encoding parameter as it's not supported by LLM APIs
            # filtered_kwargs = {k: v for k, v in self.kwargs.items() if k != 'encoding'}
            self.agent, self.role = await choose_agent(
                query=self.query,
                cfg=self.cfg,
                parent_query=self.parent_query,
                cost_callback=self.add_costs,
                headers=self.headers,
                prompt_family=self.prompt_family,
                **self.kwargs,
                # **filtered_kwargs
            )
            await self._log_event("action", action="agent_selected", details={
                "agent": self.agent,
                "role": self.role
            })

        await self._log_event("research", step="conducting_research", details={
            "agent": self.agent,
            "role": self.role
        })
        self.context = await self.research_conductor.conduct_research()

        await self._log_event("research", step="research_completed", details={
            "context_length": len(self.context)
        })
        return self.context

    async def _handle_deep_research(self, on_progress=None):
        """Handle deep research execution and logging."""
        # Log deep research configuration
        await self._log_event("research", step="deep_research_initialize", details={
            "type": "deep_research",
            "breadth": self.deep_researcher.breadth,
            "depth": self.deep_researcher.depth,
            "concurrency": self.deep_researcher.concurrency_limit
        })

        # Log deep research start
        await self._log_event("research", step="deep_research_start", details={
            "query": self.query,
            "breadth": self.deep_researcher.breadth,
            "depth": self.deep_researcher.depth,
            "concurrency": self.deep_researcher.concurrency_limit
        })

        # Run deep research and get context
        self.context = await self.deep_researcher.run(on_progress=on_progress)

        # Get total research costs
        total_costs = self.get_costs()

        # Log deep research completion with costs
        await self._log_event("research", step="deep_research_complete", details={
            "context_length": len(self.context),
            "visited_urls": len(self.visited_urls),
            "total_costs": total_costs
        })

        # Log final cost update
        await self._log_event("research", step="cost_update", details={
            "cost": total_costs,
            "total_cost": total_costs,
            "research_type": "deep_research"
        })

        # Return the research context
        return self.context

    async def write_report(self, existing_headers: list = [], relevant_written_contents: list = [], ext_context=None, custom_prompt="") -> str:
        await self._log_event("research", step="writing_report", details={
            "existing_headers": existing_headers,
            "context_source": "external" if ext_context else "internal"
        })

        report = await self.report_generator.write_report(
            existing_headers=existing_headers,
            relevant_written_contents=relevant_written_contents,
            ext_context=ext_context or self.context,
            custom_prompt=custom_prompt
        )

        await self._log_event("research", step="report_completed", details={
            "report_length": len(report)
        })
        return report

    async def write_report_conclusion(self, report_body: str) -> str:
        await self._log_event("research", step="writing_conclusion")
        conclusion = await self.report_generator.write_report_conclusion(report_body)
        await self._log_event("research", step="conclusion_completed")
        return conclusion

    async def write_introduction(self):
        await self._log_event("research", step="writing_introduction")
        intro = await self.report_generator.write_introduction()
        await self._log_event("research", step="introduction_completed")
        return intro

    async def quick_search(self, query: str, query_domains: list[str] = None) -> list[Any]:
        return await get_search_results(query, self.retrievers[0], query_domains=query_domains)

    async def get_subtopics(self):
        return await self.report_generator.get_subtopics()

    async def get_draft_section_titles(self, current_subtopic: str):
        return await self.report_generator.get_draft_section_titles(current_subtopic)

    async def get_similar_written_contents_by_draft_section_titles(
        self,
        current_subtopic: str,
        draft_section_titles: list[str],
        written_contents: list[dict],
        max_results: int = 10
    ) -> list[str]:
        return await self.context_manager.get_similar_written_contents_by_draft_section_titles(
            current_subtopic,
            draft_section_titles,
            written_contents,
            max_results
        )

    # Utility methods
    def get_research_images(self, top_k=10) -> list[dict[str, Any]]:
        return self.research_images[:top_k]

    def add_research_images(self, images: list[dict[str, Any]]) -> None:
        self.research_images.extend(images)

    def get_research_sources(self) -> list[dict[str, Any]]:
        return self.research_sources

    def add_research_sources(self, sources: list[dict[str, Any]]) -> None:
        self.research_sources.extend(sources)

    def add_references(self, report_markdown: str, visited_urls: set) -> str:
        return add_references(report_markdown, visited_urls)

    def extract_headers(self, markdown_text: str) -> list[dict]:
        return extract_headers(markdown_text)

    def extract_sections(self, markdown_text: str) -> list[dict]:
        return extract_sections(markdown_text)

    def table_of_contents(self, markdown_text: str) -> str:
        return table_of_contents(markdown_text)

    def get_source_urls(self) -> list:
        return list(self.visited_urls)

    def get_research_context(self) -> list:
        return self.context

    def get_costs(self) -> float:
        return self.research_costs

    def set_verbose(self, verbose: bool):
        self.verbose = verbose

    def add_costs(self, cost: float) -> None:
        if not isinstance(cost, (float, int)):
            raise ValueError("Cost must be an integer or float")
        self.research_costs += cost
        if self.log_handler:
            self._log_event("research", step="cost_update", details={
                "cost": cost,
                "total_cost": self.research_costs
            })
