import logging

from typing import Any

import json_repair

from gpt_researcher.config import Config
from gpt_researcher.llm_provider.generic.base import ReasoningEfforts
from gpt_researcher.prompts import PromptFamily
from gpt_researcher.utils.llm import create_chat_completion

logger = logging.getLogger(__name__)


async def get_search_results(
    query: str, retriever: Any, query_domains: list[str] | None = None, researcher=None
) -> list[dict[str, Any]]:
    """
    Retrieve search results for a given query using the provided retriever.
    
    If the retriever's class name contains "mcpretriever" (case-insensitive), the function instantiates it with the optional `researcher` argument; otherwise it instantiates the retriever without `researcher`. The function then calls the retriever's `search()` method and returns its results.
    
    Parameters:
        query: The search query string.
        query_domains: Optional list of domain strings to restrict the search.
    
    Returns:
        A list of result dictionaries as returned by the retriever's `search()` method.
    """
    # Check if this is an MCP retriever and pass the researcher instance
    if "mcpretriever" in retriever.__name__.lower():
        search_retriever = retriever(
            query,
            query_domains=query_domains,
            researcher=researcher,  # Pass researcher instance for MCP retrievers
        )
    else:
        search_retriever = retriever(query, query_domains=query_domains)

    return search_retriever.search()


async def generate_sub_queries(
    query: str,
    parent_query: str,
    report_type: str,
    context: list[dict[str, Any]],
    cfg: Config,
    cost_callback: callable | None = None,
    prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
    **kwargs,
) -> list[str]:
    """
    Generate a list of focused sub-queries for the given query by prompting an LLM.
    
    Builds a prompt via the provided PromptFamily and attempts to obtain sub-queries from the configured strategic LLM. If the initial strategic-LM call fails, it retries with the strategic token limit; if that also fails it falls back to the configured smart LLM. The final LLM response is parsed with json_repair.loads and returned.
    
    Parameters:
        query: The leaf query to decompose into sub-queries.
        parent_query: The parent/ancestor query used to provide hierarchical context.
        report_type: The report type to tailor the prompt (affects prompt content).
        context: Search results or other context passed into the prompt.
        cfg: Configuration object providing LLM models, providers, token limits, temperature, and other LLM-related settings.
        cost_callback: Optional callback invoked to report/request cost information during LLM calls.
        prompt_family: Prompt family (type or instance) used to generate the search-queries prompt.
    
    Returns:
        A list of sub-query strings parsed from the LLM response.
    
    Raises:
        Any exception from the final LLM call if all fallback attempts fail; intermediate LLM errors are handled internally and trigger retries/fallbacks.
    """
    gen_queries_prompt = prompt_family.generate_search_queries_prompt(
        query,
        parent_query,
        report_type,
        max_iterations=cfg.max_iterations or 3,
        context=context,
    )

    try:
        response = await create_chat_completion(
            model=cfg.strategic_llm_model,
            messages=[{"role": "user", "content": gen_queries_prompt}],
            llm_provider=cfg.strategic_llm_provider,
            max_tokens=None,
            llm_kwargs=cfg.llm_kwargs,
            reasoning_effort=ReasoningEfforts.Medium.value,
            cost_callback=cost_callback,
            **kwargs,
        )
    except Exception as e:
<<<<<<< HEAD
        logger.warning(f"Error with strategic LLM: {e}. Retrying with max_tokens={cfg.strategic_token_limit}.")
=======
        logger.warning(
            f"Error with strategic LLM: {e}. Retrying with max_tokens={cfg.strategic_token_limit}."
        )
>>>>>>> newdev
        logger.warning("See https://github.com/assafelovic/gpt-researcher/issues/1022")
        try:
            response = await create_chat_completion(
                model=cfg.strategic_llm_model,
                messages=[{"role": "user", "content": gen_queries_prompt}],
                max_tokens=cfg.strategic_token_limit,
                llm_provider=cfg.strategic_llm_provider,
                llm_kwargs=cfg.llm_kwargs,
                cost_callback=cost_callback,
                **kwargs,
            )
            logger.warning(
                f"Retrying with max_tokens={cfg.strategic_token_limit} successful."
            )
        except Exception as e:
            logger.warning(
                f"Retrying with max_tokens={cfg.strategic_token_limit} failed."
            )
            logger.warning(f"Error with strategic LLM: {e}. Falling back to smart LLM.")
            response = await create_chat_completion(
                model=cfg.smart_llm_model,
                messages=[{"role": "user", "content": gen_queries_prompt}],
                temperature=cfg.temperature,
                max_tokens=cfg.smart_token_limit,
                llm_provider=cfg.smart_llm_provider,
                llm_kwargs=cfg.llm_kwargs,
                cost_callback=cost_callback,
                **kwargs,
            )

    return json_repair.loads(response)


async def plan_research_outline(
    query: str,
    search_results: list[dict[str, Any]],
    agent_role_prompt: str,
    cfg: Config,
    parent_query: str,
    report_type: str,
    cost_callback: callable | None = None,
    retriever_names: list[str] | None = None,
    **kwargs,
) -> list[str]:
    """
    Decide whether to generate sub-queries for a research query and produce the research outline sub-queries.
    
    If retriever_names contains only an MCP retriever ("mcp" or "MCPRetriever"), sub-query generation is skipped and the original query is returned ([query]). If MCP is present alongside other retrievers, sub-queries are generated for the non-MCP retrieval path by delegating to generate_sub_queries.
    
    Parameters that are otherwise self-explanatory (cfg, cost_callback, agent_role_prompt, etc.) are used to configure prompt construction and LLM calls; exceptions raised by underlying calls propagate to the caller.
    
    Returns:
        List[str]: Generated sub-queries, or a single-item list containing the original query when MCP is the sole retriever.
    """
    # Handle the case where retriever_names is not provided
    if retriever_names is None:
        retriever_names = []

    # For MCP retrievers, we may want to skip sub-query generation
    # Check if MCP is the only retriever or one of multiple retrievers
<<<<<<< HEAD
    if retriever_names and ("mcp" in retriever_names or "MCPRetriever" in retriever_names):
        mcp_only = (len(retriever_names) == 1 and
                   ("mcp" in retriever_names or "MCPRetriever" in retriever_names))
=======
    if retriever_names and (
        "mcp" in retriever_names or "MCPRetriever" in retriever_names
    ):
        mcp_only = len(retriever_names) == 1 and (
            "mcp" in retriever_names or "MCPRetriever" in retriever_names
        )
>>>>>>> newdev

        if mcp_only:
            # If MCP is the only retriever, skip sub-query generation
            logger.info("Using MCP retriever only - skipping sub-query generation")
            # Return the original query to prevent additional search iterations
            return [query]
        else:
            # If MCP is one of multiple retrievers, generate sub-queries for the others
            logger.info(
                "Using MCP with other retrievers - generating sub-queries for non-MCP retrievers"
            )

    # Generate sub-queries for research outline
    sub_queries = await generate_sub_queries(
        query, parent_query, report_type, search_results, cfg, cost_callback, **kwargs
    )

    return sub_queries
