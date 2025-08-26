import asyncio
import random
import logging
import os
from ..actions.utils import stream_output
from ..actions.query_processing import plan_research_outline, get_search_results
from ..document import DocumentLoader, OnlineDocumentLoader, LangChainDocumentLoader
from ..utils.enum import ReportSource
from ..utils.logging_config import get_json_handler
from ..actions.agent_creator import choose_agent


class ResearchConductor:
    """Manages and coordinates the research process."""

    def __init__(self, researcher):
        self.researcher = researcher
        self.logger = logging.getLogger('research')
        self.json_handler = get_json_handler()
        # Add cache for MCP results to avoid redundant calls
        self._mcp_results_cache = None
        # Track MCP query count for balanced mode
        self._mcp_query_count = 0

    async def plan_research(self, query, query_domains=None):
        """
        Plan and return sub-queries (a research outline) derived from the original query.
        
        Performs an initial search to gather context, then generates a structured research outline of subtasks
        and sub-queries tailored to the researcher configuration. The function also streams brief progress
        messages (via the researcher's websocket) and logs high-level progress.
        
        Parameters:
            query (str): The main research question or prompt to decompose.
            query_domains (list[str] | None): Optional list of domain hostnames to constrain the initial search.
        
        Returns:
            list[str]: Ordered sub-queries and tasks produced for conducting the research.
        """
        await stream_output(
            "logs",
            "planning_research",
            f"🌐 Browsing the web to learn more about the task: {query}...",
            self.researcher.websocket,
        )

        search_results = await get_search_results(query, self.researcher.retrievers[0], query_domains, researcher=self.researcher)
        self.logger.info(f"Initial search results obtained: {len(search_results)} results")

        await stream_output(
            "logs",
            "planning_research",
            "🤔 Planning the research strategy and subtasks...",
            self.researcher.websocket,
        )

        retriever_names = [r.__name__ for r in self.researcher.retrievers]
        # Remove duplicate logging - this will be logged once in conduct_research instead

        outline = await plan_research_outline(
            query=query,
            search_results=search_results,
            agent_role_prompt=self.researcher.role,
            cfg=self.researcher.cfg,
            parent_query=self.researcher.parent_query,
            report_type=self.researcher.report_type,
            cost_callback=self.researcher.add_costs,
            retriever_names=retriever_names,  # Pass retriever names for MCP optimization
            **self.researcher.kwargs
        )
        self.logger.info(f"Research outline planned: {outline}")
        return outline

    async def conduct_research(self):
        """
        Run the end-to-end research workflow for the current Researcher and return the assembled context.
        
        This method:
        - Records the original query to the JSON handler (if present) and logs research start.
        - Resets per-run state (visited URLs) and selects an agent/role if not already set.
        - Determines which data sources to use based on the Researcher's report_source and orchestrates retrieval from:
          - provided source URLs,
          - web search (with optional MCP retrievers and MCP strategies),
          - local documents (including Azure or LangChain loaders),
          - hybrid combinations of local and web sources,
          - or a LangChain vector store.
        - Loads documents into the configured vector store when applicable, optionally complements URL sources with web search, and may run MCP searches according to configuration.
        - Optionally curates/ranks sources via the Researcher's source_curator if configured.
        - Streams progress and status messages when verbose, updates the JSON handler with costs and final context when available, and logs final context size.
        
        Returns:
            The final assembled research context (string or structured context as produced by the configured context managers).
        """
        if self.json_handler:
            self.json_handler.update_content("query", self.researcher.query)

        self.logger.info(f"Starting research for query: {self.researcher.query}")

        # Log active retrievers once at the start of research
        retriever_names = [r.__name__ for r in self.researcher.retrievers]
        self.logger.info(f"Active retrievers: {retriever_names}")

        # Reset visited_urls and source_urls at the start of each research task
        self.researcher.visited_urls.clear()
        research_data = []

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "starting_research",
                f"🔍 Starting the research task for '{self.researcher.query}'...",
                self.researcher.websocket,
            )
            await stream_output(
                "logs",
                "agent_generated",
                self.researcher.agent,
                self.researcher.websocket
            )

        # Choose agent and role if not already defined
        if not (self.researcher.agent and self.researcher.role):
            self.researcher.agent, self.researcher.role = await choose_agent(
                query=self.researcher.query,
                cfg=self.researcher.cfg,
                parent_query=self.researcher.parent_query,
                cost_callback=self.researcher.add_costs,
                headers=self.researcher.headers,
                prompt_family=self.researcher.prompt_family
            )

        # Check if MCP retrievers are configured
        has_mcp_retriever = any("mcpretriever" in r.__name__.lower() for r in self.researcher.retrievers)
        if has_mcp_retriever:
            self.logger.info("MCP retrievers configured and will be used with standard research flow")

        # Conduct research based on the source type
        if self.researcher.source_urls:
            self.logger.info("Using provided source URLs")
            research_data = await self._get_context_by_urls(self.researcher.source_urls)
            if research_data and len(research_data) == 0 and self.researcher.verbose:
                await stream_output(
                    "logs",
                    "answering_from_memory",
                    "🧐 I was unable to find relevant context in the provided sources...",
                    self.researcher.websocket,
                )
            if self.researcher.complement_source_urls:
                self.logger.info("Complementing with web search")
                additional_research = await self._get_context_by_web_search(self.researcher.query, [], self.researcher.query_domains)
                research_data += ' '.join(additional_research)
        elif self.researcher.report_source == ReportSource.Web.value:
            self.logger.info("Using web search with all configured retrievers")
            research_data = await self._get_context_by_web_search(self.researcher.query, [], self.researcher.query_domains)
        elif self.researcher.report_source == ReportSource.Local.value:
            self.logger.info("Using local search")
            document_data = await DocumentLoader(self.researcher.cfg.doc_path).load()
            self.logger.info(f"Loaded {len(document_data)} documents")
            if self.researcher.vector_store:
                self.researcher.vector_store.load(document_data)

            research_data = await self._get_context_by_web_search(self.researcher.query, document_data, self.researcher.query_domains)
        # Hybrid search including both local documents and web sources
        elif self.researcher.report_source == ReportSource.Hybrid.value:
            if self.researcher.document_urls:
                document_data = await OnlineDocumentLoader(self.researcher.document_urls).load()
            else:
                document_data = await DocumentLoader(self.researcher.cfg.doc_path).load()
            if self.researcher.vector_store:
                self.researcher.vector_store.load(document_data)
            docs_context = await self._get_context_by_web_search(self.researcher.query, document_data, self.researcher.query_domains)
            web_context = await self._get_context_by_web_search(self.researcher.query, [], self.researcher.query_domains)
            research_data = self.researcher.prompt_family.join_local_web_documents(docs_context, web_context)
        elif self.researcher.report_source == ReportSource.Azure.value:
            from ..document.azure_document_loader import AzureDocumentLoader
            azure_loader = AzureDocumentLoader(
                container_name=os.getenv("AZURE_CONTAINER_NAME"),
                connection_string=os.getenv("AZURE_CONNECTION_STRING")
            )
            azure_files = await azure_loader.load()
            document_data = await DocumentLoader(azure_files).load()  # Reuse existing loader
            research_data = await self._get_context_by_web_search(self.researcher.query, document_data)

        elif self.researcher.report_source == ReportSource.LangChainDocuments.value:
            langchain_documents_data = await LangChainDocumentLoader(
                self.researcher.documents
            ).load()
            if self.researcher.vector_store:
                self.researcher.vector_store.load(langchain_documents_data)
            research_data = await self._get_context_by_web_search(
                self.researcher.query, langchain_documents_data, self.researcher.query_domains
            )
        elif self.researcher.report_source == ReportSource.LangChainVectorStore.value:
            research_data = await self._get_context_by_vectorstore(self.researcher.query, self.researcher.vector_store_filter)

        # Rank and curate the sources
        self.researcher.context = research_data
        if self.researcher.cfg.curate_sources:
            self.logger.info("Curating sources")
            self.researcher.context = await self.researcher.source_curator.curate_sources(research_data)

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "research_step_finalized",
                f"Finalized research step.\n💸 Total Research Costs: ${self.researcher.get_costs()}",
                self.researcher.websocket,
            )
            if self.json_handler:
                self.json_handler.update_content("costs", self.researcher.get_costs())
                self.json_handler.update_content("context", self.researcher.context)

        self.logger.info(f"Research completed. Context size: {len(str(self.researcher.context))}")
        return self.researcher.context

    async def _get_context_by_urls(self, urls):
        """
        Scrape the provided URLs, optionally index their content into the project's vector store, and return context relevant to the current research query.
        
        Parameters:
            urls (iterable[str]): URLs to scrape and process.
        
        Returns:
            str: Aggregated context (similar content) for the researcher's current query derived from the scraped pages.
        
        Side effects:
            - Scrapes the given URLs via the researcher's scraper manager.
            - If a vector store is configured, loads the scraped content into it.
        """
        self.logger.info(f"Getting context from URLs: {urls}")

        new_search_urls = await self._get_new_urls(urls)
        self.logger.info(f"New URLs to process: {new_search_urls}")

        scraped_content = await self.researcher.scraper_manager.browse_urls(new_search_urls)
        self.logger.info(f"Scraped content from {len(scraped_content)} URLs")

        if self.researcher.vector_store:
            self.researcher.vector_store.load(scraped_content)

        context = await self.researcher.context_manager.get_similar_content_by_query(
            self.researcher.query, scraped_content
        )
        return context

    # Add logging to other methods similarly...

    async def _get_context_by_vectorstore(self, query, filter: dict | None = None):
        """
        Generates the context for the research task by searching the vectorstore
        Returns:
            context: List of context
        """
        self.logger.info(f"Starting vectorstore search for query: {query}")
        context = []
        # Generate Sub-Queries including original query
        sub_queries = await self.plan_research(query)
        # If this is not part of a sub researcher, add original query to research for better results
        if self.researcher.report_type != "subtopic_report":
            sub_queries.append(query)

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "subqueries",
                f"🗂️  I will conduct my research based on the following queries: {sub_queries}...",
                self.researcher.websocket,
                True,
                sub_queries,
            )

        # Using asyncio.gather to process the sub_queries asynchronously
        context = await asyncio.gather(
            *[
                self._process_sub_query_with_vectorstore(sub_query, filter)
                for sub_query in sub_queries
            ]
        )
        return context

    async def _get_context_by_web_search(self, query, scraped_data: list | None = None, query_domains: list | None = None):
        """
        Builds research context by performing web searches and scraping results for the given query.
        
        Performs optional MCP (multi-context provider) optimization according to the configured MCP strategy ("disabled", "fast", or "deep") and may use a cached MCP result when available. Plans sub-queries, optionally includes the original query, then processes each sub-query concurrently to collect and combine contextual summaries.
        
        Parameters:
            query (str): The main research query.
            scraped_data (list | None): Pre-scraped content to include in sub-query processing (defaults to empty list).
            query_domains (list | None): Optional list of domains to constrain searches.
        
        Returns:
            str | list: A combined context string when any sub-query yields content; otherwise an empty list. On unexpected errors the function logs the error and returns an empty list.
        """
        self.logger.info(f"Starting web search for query: {query}")

        if scraped_data is None:
            scraped_data = []
        if query_domains is None:
            query_domains = []

        # **CONFIGURABLE MCP OPTIMIZATION: Control MCP strategy**
        mcp_retrievers = [r for r in self.researcher.retrievers if "mcpretriever" in r.__name__.lower()]

        # Get MCP strategy configuration
        mcp_strategy = self._get_mcp_strategy()

        if mcp_retrievers and self._mcp_results_cache is None:
            if mcp_strategy == "disabled":
                # MCP disabled - skip MCP research entirely
                self.logger.info("MCP disabled by strategy, skipping MCP research")
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "mcp_disabled",
                        "⚡ MCP research disabled by configuration",
                        self.researcher.websocket,
                    )
            elif mcp_strategy == "fast":
                # Fast: Run MCP once with original query
                self.logger.info("MCP fast strategy: Running once with original query")
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "mcp_optimization",
                        "🚀 MCP Fast: Running once for main query (performance mode)",
                        self.researcher.websocket,
                    )

                # Execute MCP research once with the original query
                mcp_context = await self._execute_mcp_research_for_queries([query], mcp_retrievers)
                self._mcp_results_cache = mcp_context
                self.logger.info(f"MCP results cached: {len(mcp_context)} total context entries")
            elif mcp_strategy == "deep":
                # Deep: Will run MCP for all queries (original behavior) - defer to per-query execution
                self.logger.info("MCP deep strategy: Will run for all queries")
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "mcp_comprehensive",
                        "🔍 MCP Deep: Will run for each sub-query (thorough mode)",
                        self.researcher.websocket,
                    )
                # Don't cache - let each sub-query run MCP individually
            else:
                # Unknown strategy - default to fast
                self.logger.warning(f"Unknown MCP strategy '{mcp_strategy}', defaulting to fast")
                mcp_context = await self._execute_mcp_research_for_queries([query], mcp_retrievers)
                self._mcp_results_cache = mcp_context
                self.logger.info(f"MCP results cached: {len(mcp_context)} total context entries")

        # Generate Sub-Queries including original query
        sub_queries = await self.plan_research(query, query_domains)
        self.logger.info(f"Generated sub-queries: {sub_queries}")

        # If this is not part of a sub researcher, add original query to research for better results
        if self.researcher.report_type != "subtopic_report":
            sub_queries.append(query)

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "subqueries",
                f"🗂️ I will conduct my research based on the following queries: {sub_queries}...",
                self.researcher.websocket,
                True,
                sub_queries,
            )

        # Using asyncio.gather to process the sub_queries asynchronously
        try:
            context = await asyncio.gather(
                *[
                    self._process_sub_query(sub_query, scraped_data, query_domains)
                    for sub_query in sub_queries
                ]
            )
            self.logger.info(f"Gathered context from {len(context)} sub-queries")
            # Filter out empty results and join the context
            context = [c for c in context if c]
            if context:
                combined_context = " ".join(context)
                self.logger.info(f"Combined context size: {len(combined_context)}")
                return combined_context
            return []
        except Exception as e:
            self.logger.error(f"Error during web search: {e}", exc_info=True)
            return []

    def _get_mcp_strategy(self) -> str:
        """
        Get the MCP strategy configuration.
        
        Priority:
        1. Instance-level setting (self.researcher.mcp_strategy)
        2. Config file setting (self.researcher.cfg.mcp_strategy) 
        3. Default value ("fast")
        
        Returns:
            str: MCP strategy
                "disabled" = Skip MCP entirely
                "fast" = Run MCP once with original query (default)
                "deep" = Run MCP for all sub-queries
        """
        # Check instance-level setting first
        if hasattr(self.researcher, 'mcp_strategy') and self.researcher.mcp_strategy is not None:
            return self.researcher.mcp_strategy

        # Check config setting
        if hasattr(self.researcher.cfg, 'mcp_strategy'):
            return self.researcher.cfg.mcp_strategy

        # Default to fast mode
        return "fast"

    async def _execute_mcp_research_for_queries(self, queries: list, mcp_retrievers: list) -> list:
        """
        Execute MCP searches for each query across the provided MCP retrievers and aggregate results.
        
        For each query, instantiates/runs each retriever via _execute_mcp_research and collects non-empty results into context entries of the form:
            {"content": str, "url": str, "title": str, "query": str, "source_type": "mcp"}
        
        Side effects:
        - Logs progress and per-query counts.
        - When researcher.verbose is true, streams status messages to the researcher's websocket.
        - Exceptions from individual retrievers are caught, logged, and do not abort the overall operation.
        
        Parameters:
            queries (list): Queries to run MCP research for.
            mcp_retrievers (list): Iterable of MCP retriever classes or factory callables used by _execute_mcp_research.
        
        Returns:
            list: Aggregated list of MCP context entries (one dict per result).
        """
        all_mcp_context = []

        for i, query in enumerate(queries, 1):
            self.logger.info(f"Executing MCP research for query {i}/{len(queries)}: {query}")

            for retriever in mcp_retrievers:
                try:
                    mcp_results = await self._execute_mcp_research(retriever, query)
                    if mcp_results:
                        for result in mcp_results:
                            content = result.get("body", "")
                            url = result.get("href", "")
                            title = result.get("title", "")

                            if content:
                                context_entry = {
                                    "content": content,
                                    "url": url,
                                    "title": title,
                                    "query": query,
                                    "source_type": "mcp"
                                }
                                all_mcp_context.append(context_entry)

                        self.logger.info(f"Added {len(mcp_results)} MCP results for query: {query}")

                        if self.researcher.verbose:
                            await stream_output(
                                "logs",
                                "mcp_results_cached",
                                f"✅ Cached {len(mcp_results)} MCP results from query {i}/{len(queries)}",
                                self.researcher.websocket,
                            )
                except Exception as e:
                    self.logger.error(f"Error in MCP research for query '{query}': {e}")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_cache_error",
                            f"⚠️ MCP research error for query {i}, continuing with other sources",
                            self.researcher.websocket,
                        )

        return all_mcp_context

    async def _process_sub_query(self, sub_query: str, scraped_data: list = [], query_domains: list = []):
        """
        Run research for a single sub-query: gather scraped web content, optionally run MCP retrievers, and combine results into a single context string.
        
        This performs these steps:
        - Optionally reuses provided `scraped_data`; if not provided, scrapes source URLs discovered for the sub-query.
        - Runs configured MCP retrievers according to the MCP strategy ("disabled", "fast", "deep"), possibly reusing a cached MCP result set or executing per-query MCP searches.
        - Retrieves content similar to the sub-query from scraped results via the researcher's context manager.
        - Merges MCP entries and web-derived content into a single formatted context string.
        
        Parameters:
            sub_query (str): The sub-query to research.
            scraped_data (list, optional): Pre-scraped content to use instead of performing a new scrape. If empty, the function will discover and scrape URLs. Defaults to [].
            query_domains (list, optional): Optional list of domain constraints used when searching/scraping sources.
        
        Returns:
            str: Combined research context for the sub-query. Returns an empty string on error or when no content is found.
        
        Side effects:
        - May log events via the instance logger and json_handler.
        - May send progress/status messages over the researcher's websocket when verbose.
        - May invoke MCP retrievers and the context manager, and may trigger scraping and vector store operations.
        """
        if self.json_handler:
            self.json_handler.log_event("sub_query", {
                "query": sub_query,
                "scraped_data_size": len(scraped_data)
            })

        if self.researcher.verbose:
            await stream_output(
                "logs",
                "running_subquery_research",
                f"\n🔍 Running research for '{sub_query}'...",
                self.researcher.websocket,
            )

        try:
            # Identify MCP retrievers
            mcp_retrievers = [r for r in self.researcher.retrievers if "mcpretriever" in r.__name__.lower()]
            non_mcp_retrievers = [r for r in self.researcher.retrievers if "mcpretriever" not in r.__name__.lower()]

            # Initialize context components
            mcp_context = []
            web_context = ""

            # Get MCP strategy configuration
            mcp_strategy = self._get_mcp_strategy()

            # **CONFIGURABLE MCP PROCESSING**
            if mcp_retrievers:
                if mcp_strategy == "disabled":
                    # MCP disabled - skip entirely
                    self.logger.info(f"MCP disabled for sub-query: {sub_query}")
                elif mcp_strategy == "fast" and self._mcp_results_cache is not None:
                    # Fast: Use cached results
                    mcp_context = self._mcp_results_cache.copy()

                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_cache_reuse",
                            f"♻️ Reusing cached MCP results ({len(mcp_context)} sources) for: {sub_query}",
                            self.researcher.websocket,
                        )

                    self.logger.info(f"Reused {len(mcp_context)} cached MCP results for sub-query: {sub_query}")
                elif mcp_strategy == "deep":
                    # Deep: Run MCP for every sub-query
                    self.logger.info(f"Running deep MCP research for: {sub_query}")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_comprehensive_run",
                            f"🔍 Running deep MCP research for: {sub_query}",
                            self.researcher.websocket,
                        )

                    mcp_context = await self._execute_mcp_research_for_queries([sub_query], mcp_retrievers)
                else:
                    # Fallback: if no cache and not deep mode, run MCP for this query
                    self.logger.warning("MCP cache not available, falling back to per-sub-query execution")
                    if self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_fallback",
                            f"🔌 MCP cache unavailable, running MCP research for: {sub_query}",
                            self.researcher.websocket,
                        )

                    mcp_context = await self._execute_mcp_research_for_queries([sub_query], mcp_retrievers)

            # Get web search context using non-MCP retrievers (if no scraped data provided)
            if not scraped_data:
                scraped_data = await self._scrape_data_by_urls(sub_query, query_domains)
                self.logger.info(f"Scraped data size: {len(scraped_data)}")

            # Get similar content based on scraped data
            if scraped_data:
                web_context = await self.researcher.context_manager.get_similar_content_by_query(sub_query, scraped_data)
                self.logger.info(f"Web content found for sub-query: {len(str(web_context)) if web_context else 0} chars")

            # Combine MCP context with web context intelligently
            combined_context = self._combine_mcp_and_web_context(mcp_context, web_context, sub_query)

            # Log context combination results
            if combined_context:
                context_length = len(str(combined_context))
                self.logger.info(f"Combined context for '{sub_query}': {context_length} chars")

                if self.researcher.verbose:
                    mcp_count = len(mcp_context)
                    web_available = bool(web_context)
                    cache_used = self._mcp_results_cache is not None and mcp_retrievers and mcp_strategy != "deep"
                    cache_status = " (cached)" if cache_used else ""
                    await stream_output(
                        "logs",
                        "context_combined",
                        f"📚 Combined research context: {mcp_count} MCP sources{cache_status}, {'web content' if web_available else 'no web content'}",
                        self.researcher.websocket,
                    )
            else:
                self.logger.warning(f"No combined context found for sub-query: {sub_query}")
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "subquery_context_not_found",
                        f"🤷 No content found for '{sub_query}'...",
                        self.researcher.websocket,
                    )

            if combined_context and self.json_handler:
                self.json_handler.log_event("content_found", {
                    "sub_query": sub_query,
                    "content_size": len(str(combined_context)),
                    "mcp_sources": len(mcp_context),
                    "web_content": bool(web_context)
                })

            return combined_context

        except Exception as e:
            self.logger.error(f"Error processing sub-query {sub_query}: {e}", exc_info=True)
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "subquery_error",
                    f"❌ Error processing '{sub_query}': {e!s}",
                    self.researcher.websocket,
                )
            return ""

    async def _execute_mcp_research(self, retriever, query):
        """
        Run MCP (multi-channel/provider) research for a single retriever and query using the two-stage approach.
        
        This asynchronously instantiates the provided MCP retriever class with the current researcher context, executes its search (bounded by researcher.cfg.max_search_results_per_query), and returns the retriever's results. If the researcher is in verbose mode, progress and outcomes are streamed to the researcher's websocket. Any errors are caught and result in an empty list.
        
        Parameters:
            retriever (type): MCP retriever class (callable) that will be instantiated with
                parameters including `query`, researcher headers, query_domains, websocket, and the
                full researcher instance.
            query (str): The search query to run against the MCP retriever.
        
        Returns:
            list: A list of search result items returned by the retriever, or an empty list if no results
            were found or an error occurred.
        """
        retriever_name = retriever.__name__

        self.logger.info(f"Executing MCP research with {retriever_name} for query: {query}")

        try:
            # Instantiate the MCP retriever with proper parameters
            # Pass the researcher instance (self.researcher) which contains both cfg and mcp_configs
            retriever_instance = retriever(
                query=query,
                headers=self.researcher.headers,
                query_domains=self.researcher.query_domains,
                websocket=self.researcher.websocket,
                researcher=self.researcher  # Pass the entire researcher instance
            )

            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "mcp_retrieval_stage1",
                    f"🧠 Stage 1: Selecting optimal MCP tools for: {query}",
                    self.researcher.websocket,
                )

            # Execute the two-stage MCP search
            results = retriever_instance.search(
                max_results=self.researcher.cfg.max_search_results_per_query
            )

            if results:
                result_count = len(results)
                self.logger.info(f"MCP research completed: {result_count} results from {retriever_name}")

                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "mcp_research_complete",
                        f"🎯 MCP research completed: {result_count} intelligent results obtained",
                        self.researcher.websocket,
                    )

                return results
            else:
                self.logger.info(f"No results returned from MCP research with {retriever_name}")
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "mcp_no_results",
                        f"ℹ️ No relevant information found via MCP for: {query}",
                        self.researcher.websocket,
                    )
                return []

        except Exception as e:
            self.logger.error(f"Error in MCP research with {retriever_name}: {e!s}")
            if self.researcher.verbose:
                await stream_output(
                    "logs",
                    "mcp_research_error",
                    f"⚠️ MCP research error: {e!s} - continuing with other sources",
                    self.researcher.websocket,
                )
            return []

    def _combine_mcp_and_web_context(self, mcp_context: list, web_context: str, sub_query: str) -> str:
        """
        Combine web-derived context and MCP (multi-connector pipeline) context entries into a single research context string.
        
        The function appends non-empty web_context first, then appends formatted MCP entries. Each MCP entry includes its content followed by a citation line containing the title and, when available and not an internal LLM analysis marker, the source URL. Multiple MCP entries are separated by a visual delimiter ("---"). If neither web nor MCP content is present, an empty string is returned.
        
        Parameters:
            mcp_context (list): List of MCP context entries (dicts expected to contain at least "content"; optional "url" and "title").
            web_context (str): Context aggregated from web search/scraping for the same sub-query.
            sub_query (str): The sub-query for which the combined context is being produced (used for logging).
        
        Returns:
            str: The combined context string (web context followed by formatted MCP section), or an empty string when no context is available.
        """
        combined_parts = []

        # Add web context first if available
        if web_context and web_context.strip():
            combined_parts.append(web_context.strip())
            self.logger.debug(f"Added web context: {len(web_context)} chars")

        # Add MCP context with proper formatting
        if mcp_context:
            mcp_formatted = []

            for i, item in enumerate(mcp_context):
                content = item.get("content", "")
                url = item.get("url", "")
                title = item.get("title", f"MCP Result {i+1}")

                if content and content.strip():
                    # Create a well-formatted context entry
                    if url and url != "mcp://llm_analysis":
                        citation = f"\n\n*Source: {title} ({url})*"
                    else:
                        citation = f"\n\n*Source: {title}*"

                    formatted_content = f"{content.strip()}{citation}"
                    mcp_formatted.append(formatted_content)

            if mcp_formatted:
                # Join MCP results with clear separation
                mcp_section = "\n\n---\n\n".join(mcp_formatted)
                combined_parts.append(mcp_section)
                self.logger.debug(f"Added {len(mcp_context)} MCP context entries")

        # Combine all parts
        if combined_parts:
            final_context = "\n\n".join(combined_parts)
            self.logger.info(f"Combined context for '{sub_query}': {len(final_context)} total chars")
            return final_context
        else:
            self.logger.warning(f"No context to combine for sub-query: {sub_query}")
            return ""

    async def _process_sub_query_with_vectorstore(self, sub_query: str, filter: dict | None = None):
        """Takes in a sub query and gathers context from the user provided vector store

        Args:
            sub_query (str): The sub-query generated from the original query

        Returns:
            str: The context gathered from search
        """
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "running_subquery_with_vectorstore_research",
                f"\n🔍 Running research for '{sub_query}'...",
                self.researcher.websocket,
            )

        context = await self.researcher.context_manager.get_similar_content_by_query_with_vectorstore(sub_query, filter)

        return context

    async def _get_new_urls(self, url_set_input):
        """Gets the new urls from the given url set.
        Args: url_set_input (set[str]): The url set to get the new urls from
        Returns: list[str]: The new urls from the given url set
        """

        new_urls = []
        for url in url_set_input:
            if url not in self.researcher.visited_urls:
                self.researcher.visited_urls.add(url)
                new_urls.append(url)
                if self.researcher.verbose:
                    await stream_output(
                        "logs",
                        "added_source_url",
                        f"✅ Added source url to research: {url}\n",
                        self.researcher.websocket,
                        True,
                        url,
                    )

        return new_urls

    async def _search_relevant_source_urls(self, query, query_domains: list | None = None):
        """
        Finds and returns a shuffled list of new source URLs relevant to `query` by running each non-MCP retriever.
        
        This method iterates over the researcher's configured retriever classes (skipping any whose name indicates an MCP retriever), instantiates each retriever for the given query, performs the retriever search (executed on a worker thread), extracts hrefs from results, deduplicates against already visited URLs, shuffles the final list, and returns the new URLs to process. Retriever errors are logged and do not stop processing.
        
        Parameters:
            query (str): The query or sub-query to search for.
            query_domains (list[str] | None): Optional list of domains to constrain retrievers' searches.
        
        Returns:
            list[str]: A shuffled list of new, deduplicated URLs discovered for the query.
        """
        new_search_urls = []
        if query_domains is None:
            query_domains = []

        # Iterate through the currently set retrievers
        # This allows the method to work when retrievers are temporarily modified
        for retriever_class in self.researcher.retrievers:
            # Skip MCP retrievers as they don't provide URLs for scraping
            if "mcpretriever" in retriever_class.__name__.lower():
                continue

            try:
                # Instantiate the retriever with the sub-query
                retriever = retriever_class(query, query_domains=query_domains)

                # Perform the search using the current retriever
                search_results = await asyncio.to_thread(
                    retriever.search, max_results=self.researcher.cfg.max_search_results_per_query
                )

                # Collect new URLs from search results
                search_urls = [url.get("href") for url in search_results if url.get("href")]
                new_search_urls.extend(search_urls)
            except Exception as e:
                self.logger.error(f"Error searching with {retriever_class.__name__}: {e}")

        # Get unique URLs
        new_search_urls = await self._get_new_urls(new_search_urls)
        random.shuffle(new_search_urls)

        return new_search_urls

    async def _scrape_data_by_urls(self, sub_query, query_domains: list | None = None):
        """
        Search retrievers for URLs relevant to a sub-query, scrape those URLs, and return the scraped content.
        
        Performs a retriever search for `sub_query` (optionally restricted to `query_domains`), scrapes the resulting URLs via the scraper manager, and—if a vector store is configured—loads the scraped content into it. When verbose mode is enabled, progress messages are streamed to the researcher's websocket.
        
        Parameters:
            sub_query (str): The sub-query to search for.
            query_domains (list | None): Optional list of domain strings to restrict the search to; defaults to [].
        
        Returns:
            list: Scraped content entries returned by the scraper manager (typically a list of content dicts/objects).
        """
        if query_domains is None:
            query_domains = []

        new_search_urls = await self._search_relevant_source_urls(sub_query, query_domains)

        # Log the research process if verbose mode is on
        if self.researcher.verbose:
            await stream_output(
                "logs",
                "researching",
                "🤔 Researching for relevant information across multiple sources...\n",
                self.researcher.websocket,
            )

        # Scrape the new URLs
        scraped_content = await self.researcher.scraper_manager.browse_urls(new_search_urls)

        if self.researcher.vector_store:
            self.researcher.vector_store.load(scraped_content)

        return scraped_content

    async def _search(self, retriever, query):
        """
        Search using the provided retriever class for the given query and return its results.
        
        This function instantiates the retriever (passing query, headers, domains and, for MCP retrievers, websocket and researcher), calls its `search` method with the configured max results, and returns the list of results. If the retriever is an MCP retriever and verbose mode is enabled, progress and result summaries may be streamed to the researcher's websocket. On error or if the retriever lacks a `search` method, an empty list is returned.
        
        Parameters:
            retriever (type): Retriever class to instantiate and call; expected to expose a synchronous `search(max_results=...)` method on instances.
            query (str): Search query to pass to the retriever.
        
        Returns:
            list: A list of search result dicts (may be empty on no results or error).
        """
        retriever_name = retriever.__name__
        is_mcp_retriever = "mcpretriever" in retriever_name.lower()

        self.logger.info(f"Searching with {retriever_name} for query: {query}")

        try:
            # Instantiate the retriever
            retriever_instance = retriever(
                query=query,
                headers=self.researcher.headers,
                query_domains=self.researcher.query_domains,
                websocket=self.researcher.websocket if is_mcp_retriever else None,
                researcher=self.researcher if is_mcp_retriever else None
            )

            # Log MCP server configurations if using MCP retriever
            if is_mcp_retriever and self.researcher.verbose:
                await stream_output(
                    "logs",
                    "mcp_retrieval",
                    f"🔌 Consulting MCP server(s) for information on: {query}",
                    self.researcher.websocket,
                )

            # Perform the search
            if hasattr(retriever_instance, 'search'):
                results = retriever_instance.search(
                    max_results=self.researcher.cfg.max_search_results_per_query
                )

                # Log result information
                if results:
                    result_count = len(results)
                    self.logger.info(f"Received {result_count} results from {retriever_name}")

                    # Special logging for MCP retriever
                    if is_mcp_retriever:
                        if self.researcher.verbose:
                            await stream_output(
                                "logs",
                                "mcp_results",
                                f"✓ Retrieved {result_count} results from MCP server",
                                self.researcher.websocket,
                            )

                        # Log result details
                        for i, result in enumerate(results[:3]):  # Log first 3 results
                            title = result.get("title", "No title")
                            url = result.get("href", "No URL")
                            content_length = len(result.get("body", "")) if result.get("body") else 0
                            self.logger.info(f"MCP result {i+1}: '{title}' from {url} ({content_length} chars)")

                        if result_count > 3:
                            self.logger.info(f"... and {result_count - 3} more MCP results")
                else:
                    self.logger.info(f"No results returned from {retriever_name}")
                    if is_mcp_retriever and self.researcher.verbose:
                        await stream_output(
                            "logs",
                            "mcp_no_results",
                            f"ℹ️ No relevant information found from MCP server for: {query}",
                            self.researcher.websocket,
                        )

                return results
            else:
                self.logger.error(f"Retriever {retriever_name} does not have a search method")
                return []
        except Exception as e:
            self.logger.error(f"Error searching with {retriever_name}: {e!s}")
            if is_mcp_retriever and self.researcher.verbose:
                await stream_output(
                    "logs",
                    "mcp_error",
                    f"❌ Error retrieving information from MCP server: {e!s}",
                    self.researcher.websocket,
                )
            return []

    async def _extract_content(self, results):
        """
        Extract and return scraped content from search result entries.
        
        Parses `results` for entries containing an "href" key, filters out URLs already present in
        self.researcher.visited_urls, scrapes the remaining URLs via self.researcher.scraper_manager.browse_urls,
        marks those URLs as visited, and returns the scraped content list.
        
        Parameters:
            results (iterable): Search result entries (expected to be dict-like objects with an "href" key).
        
        Returns:
            list: Scraped content objects for newly visited URLs (empty list if no new URLs found).
        
        Side effects:
            - Calls self.researcher.scraper_manager.browse_urls(...) asynchronously.
            - Updates self.researcher.visited_urls with the newly visited URLs.
        """
        self.logger.info(f"Extracting content from {len(results)} search results")

        # Get the URLs from the search results
        urls = []
        for result in results:
            if isinstance(result, dict) and "href" in result:
                urls.append(result["href"])

        # Skip if no URLs found
        if not urls:
            return []

        # Make sure we don't visit URLs we've already visited
        new_urls = [url for url in urls if url not in self.researcher.visited_urls]

        # Return empty if no new URLs
        if not new_urls:
            return []

        # Scrape the content from the URLs
        scraped_content = await self.researcher.scraper_manager.browse_urls(new_urls)

        # Add the URLs to visited_urls
        self.researcher.visited_urls.update(new_urls)

        return scraped_content

    async def _summarize_content(self, query, content):
        """
        Summarize extracted content relative to a query using the researcher's context manager.
        
        If `content` is empty or falsy, returns an empty string. Otherwise delegates to
        self.researcher.context_manager.get_similar_content_by_query to produce a focused
        summary for the provided query.
        
        Parameters:
            query (str): The query used to focus the summary.
            content (str | Sequence[str] | Sequence[dict]): Extracted content to summarize
                (single text, list of texts, or list of document-like records).
        
        Returns:
            str: The summarized content for the query, or an empty string when `content` is empty.
        """
        self.logger.info(f"Summarizing content for query: {query}")

        # Skip if no content
        if not content:
            return ""

        # Summarize the content using the context manager
        summary = await self.researcher.context_manager.get_similar_content_by_query(
            query, content
        )

        return summary

    async def _update_search_progress(self, current, total):
        """
        Emit an updated research progress percentage to the active websocket when verbose.
        
        Asynchronously computes progress as int((current / total) * 100) and streams a progress message
        to the researcher's websocket if verbosity is enabled and a websocket is present.
        
        Parameters:
            current (int): Number of sub-queries processed so far.
            total (int): Total number of sub-queries; must be > 0 to avoid a division error.
        """
        if self.researcher.verbose and self.researcher.websocket:
            progress = int((current / total) * 100)
            await stream_output(
                "logs",
                "research_progress",
                f"📊 Research Progress: {progress}%",
                self.researcher.websocket,
                True,
                {
                    "current": current,
                    "total": total,
                    "progress": progress
                }
            )

