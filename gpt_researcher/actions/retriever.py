def get_retriever(retriever: str):
    """
    Gets the retriever
    Args:
        retriever (str): retriever name

    Returns:
        retriever: Retriever class

    """
    match retriever:
        case "google":
            from gpt_researcher.retrievers import GoogleSearch

            return GoogleSearch
        case "searx":
            from gpt_researcher.retrievers import SearxSearch

            return SearxSearch
        case "searchapi":
            from gpt_researcher.retrievers import SearchApiSearch

            return SearchApiSearch
        case "serpapi":
            from gpt_researcher.retrievers import SerpApiSearch

            return SerpApiSearch
        case "serper":
            from gpt_researcher.retrievers import SerperSearch

            return SerperSearch
        case "duckduckgo":
            from gpt_researcher.retrievers import Duckduckgo

            return Duckduckgo
        case "bing":
            from gpt_researcher.retrievers import BingSearch

            return BingSearch
        case "arxiv":
            from gpt_researcher.retrievers import ArxivSearch

            return ArxivSearch
        case "tavily":
            from gpt_researcher.retrievers import TavilySearch

            return TavilySearch
        case "exa":
            from gpt_researcher.retrievers import ExaSearch

            return ExaSearch
        case "semantic_scholar":
            from gpt_researcher.retrievers import SemanticScholarSearch

            return SemanticScholarSearch
        case "pubmed_central":
            from gpt_researcher.retrievers import PubMedCentralSearch

            return PubMedCentralSearch
        case "custom":
            from gpt_researcher.retrievers import CustomRetriever

            return CustomRetriever
        case "mcp":
            from gpt_researcher.retrievers import MCPRetriever

            return MCPRetriever

        case _:
            return None


def get_retrievers(headers: dict[str, str], cfg):
    """
    Select retriever classes based on request headers, configuration, or a built-in default.
    
    Checks headers first (supports "retrievers" as a comma-separated string or "retriever" as a single value), then the cfg object (cfg.retrievers may be a comma-separated string or an iterable, cfg.retriever as a single value). If none are provided, the default retriever name is used. Each resolved retriever name is mapped to a retriever class via get_retriever(); any unrecognized name is replaced with the default retriever class.
    
    Parameters:
        headers (dict[str, str]): HTTP-like headers that may contain "retrievers" or "retriever".
        cfg: Configuration object that may provide `retrievers` (str or iterable) or `retriever` (str).
    
    Returns:
        list: A list of retriever classes (not instances). Unrecognized names are substituted with the default retriever class.
    """
    # Check headers first for multiple retrievers
    if headers.get("retrievers"):
        retrievers = headers.get("retrievers").split(",")
    # If not found, check headers for a single retriever
    elif headers.get("retriever"):
        retrievers = [headers.get("retriever")]
    # If not in headers, check config for multiple retrievers
    elif cfg.retrievers:
        # Handle both list and string formats for config retrievers
        if isinstance(cfg.retrievers, str):
            retrievers = cfg.retrievers.split(",")
        else:
            retrievers = cfg.retrievers
        # Strip whitespace from each retriever name
        retrievers = [r.strip() for r in retrievers]
    # If not found, check config for a single retriever
    elif cfg.retriever:
        retrievers = [cfg.retriever]
    # If still not set, use default retriever
    else:
        retrievers = [get_default_retriever().__name__]

    # Convert retriever names to actual retriever classes
    # Use get_default_retriever() as a fallback for any invalid retriever names
<<<<<<< HEAD
    retriever_classes = [
        get_retriever(r) or get_default_retriever() for r in retrievers
    ]
=======
    retriever_classes = [get_retriever(r) or get_default_retriever() for r in retrievers]
>>>>>>> 1027e1d0 (Fix linting issues)

    return retriever_classes


def get_default_retriever():
    from gpt_researcher.retrievers import TavilySearch

    return TavilySearch
