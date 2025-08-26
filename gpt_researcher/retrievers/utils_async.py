"""Async-friendly utilities for retrievers with proper HTTP client usage."""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional
import aiohttp
import httpx

logger = logging.getLogger(__name__)


async def jina_rerank_async(
    query: str, 
    documents: List[Dict[str, Any]], 
    top_n: Optional[int] = None,
    model: str = "jina-reranker-v2-base-multilingual",
    session: Optional[aiohttp.ClientSession] = None
) -> List[Dict[str, Any]]:
    """
    Rerank documents using Jina AI reranker API with proper async handling.
    
    Args:
        query: The search query to rerank documents against
        documents: List of document dictionaries to rerank
        top_n: Maximum number of documents to return
        model: Jina reranker model to use
        session: Optional aiohttp session for connection reuse
        
    Returns:
        Reranked list of documents, or original documents if reranking fails
        
    Note:
        Get your Jina AI API key for free: https://jina.ai/?sui=apikey
    """
    # Early return for empty documents
    if not documents:
        logger.debug("No documents to rerank")
        return []
    
    api_key = os.environ.get("JINA_API_KEY")
    if not api_key:
        logger.debug("No JINA_API_KEY found, returning original documents")
        return documents[:top_n] if top_n else documents
    
    try:
        # Extract text content from documents
        texts = []
        for doc in documents:
            content = (
                doc.get("body") 
                or doc.get("content") 
                or doc.get("snippet") 
                or doc.get("text")
                or ""
            )
            texts.append(content)
        
        # Skip if all texts are empty
        if not any(texts):
            logger.warning("All documents have empty content, returning original list")
            return documents[:top_n] if top_n else documents
        
        # Prepare request payload
        payload = {
            "model": model,
            "query": query,
            "documents": texts,
        }
        
        if top_n:
            payload["top_n"] = min(top_n, len(documents))
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        
        # Use provided session or create a new one
        if session:
            async with session.post(
                "https://api.jina.ai/v1/rerank",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                response.raise_for_status()
                json_data = await response.json()
        else:
            # Create a new session for this request
            async with aiohttp.ClientSession() as new_session:
                async with new_session.post(
                    "https://api.jina.ai/v1/rerank",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    response.raise_for_status()
                    json_data = await response.json()
        
        # Process results
        results = json_data.get("results", [])
        if not results:
            logger.warning("Jina rerank returned no results")
            return documents[:top_n] if top_n else documents
        
        # Map results back to original documents
        reranked = []
        for result in results:
            idx = result.get("index")
            if idx is not None and 0 <= idx < len(documents):
                # Add relevance score to document for potential later use
                doc = documents[idx].copy()
                doc["relevance_score"] = result.get("relevance_score", 0)
                reranked.append(doc)
        
        # If reranking produced fewer results than expected, log it
        if len(reranked) < len(documents):
            logger.info(f"Reranking returned {len(reranked)} of {len(documents)} documents")
        
        return reranked if reranked else documents[:top_n] if top_n else documents
        
    except aiohttp.ClientError as e:
        logger.error(f"Jina rerank network error: {e}")
        return documents[:top_n] if top_n else documents
    except Exception as e:
        logger.error(f"Jina rerank failed: {e}")
        return documents[:top_n] if top_n else documents


async def jina_rerank_httpx(
    query: str, 
    documents: List[Dict[str, Any]], 
    top_n: Optional[int] = None,
    model: str = "jina-reranker-v2-base-multilingual",
    client: Optional[httpx.AsyncClient] = None
) -> List[Dict[str, Any]]:
    """
    Alternative implementation using httpx for Jina reranking.
    
    Args:
        query: The search query to rerank documents against
        documents: List of document dictionaries to rerank
        top_n: Maximum number of documents to return
        model: Jina reranker model to use
        client: Optional httpx async client for connection reuse
        
    Returns:
        Reranked list of documents, or original documents if reranking fails
        
    Note:
        Get your Jina AI API key for free: https://jina.ai/?sui=apikey
    """
    # Early return for empty documents
    if not documents:
        logger.debug("No documents to rerank")
        return []
    
    api_key = os.environ.get("JINA_API_KEY")
    if not api_key:
        logger.debug("No JINA_API_KEY found, returning original documents")
        return documents[:top_n] if top_n else documents
    
    try:
        # Extract text content from documents
        texts = []
        for doc in documents:
            content = (
                doc.get("body") 
                or doc.get("content") 
                or doc.get("snippet") 
                or doc.get("text")
                or ""
            )
            texts.append(content)
        
        # Skip if all texts are empty
        if not any(texts):
            logger.warning("All documents have empty content, returning original list")
            return documents[:top_n] if top_n else documents
        
        # Prepare request payload
        payload = {
            "model": model,
            "query": query,
            "documents": texts,
        }
        
        if top_n:
            payload["top_n"] = min(top_n, len(documents))
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        
        # Use provided client or create a new one
        if client:
            response = await client.post(
                "https://api.jina.ai/v1/rerank",
                json=payload,
                headers=headers,
                timeout=10.0
            )
        else:
            # Create a new client for this request
            async with httpx.AsyncClient() as new_client:
                response = await new_client.post(
                    "https://api.jina.ai/v1/rerank",
                    json=payload,
                    headers=headers,
                    timeout=10.0
                )
        
        response.raise_for_status()
        json_data = response.json()
        
        # Process results
        results = json_data.get("results", [])
        if not results:
            logger.warning("Jina rerank returned no results")
            return documents[:top_n] if top_n else documents
        
        # Map results back to original documents
        reranked = []
        for result in results:
            idx = result.get("index")
            if idx is not None and 0 <= idx < len(documents):
                # Add relevance score to document for potential later use
                doc = documents[idx].copy()
                doc["relevance_score"] = result.get("relevance_score", 0)
                reranked.append(doc)
        
        # If reranking produced fewer results than expected, log it
        if len(reranked) < len(documents):
            logger.info(f"Reranking returned {len(reranked)} of {len(documents)} documents")
        
        return reranked if reranked else documents[:top_n] if top_n else documents
        
    except httpx.HTTPError as e:
        logger.error(f"Jina rerank network error: {e}")
        return documents[:top_n] if top_n else documents
    except Exception as e:
        logger.error(f"Jina rerank failed: {e}")
        return documents[:top_n] if top_n else documents


def jina_rerank_sync_wrapper(
    query: str,
    documents: List[Dict[str, Any]],
    top_n: Optional[int] = None,
    model: str = "jina-reranker-v2-base-multilingual"
) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for jina_rerank_async to maintain backward compatibility.
    
    This function detects if it's being called from an async context and handles it appropriately.
    
    Args:
        query: The search query to rerank documents against
        documents: List of document dictionaries to rerank
        top_n: Maximum number of documents to return
        model: Jina reranker model to use
        
    Returns:
        Reranked list of documents, or original documents if reranking fails
    """
    try:
        # Check if we're in an async context
        loop = asyncio.get_running_loop()
        # We're in an async context, we shouldn't be called directly
        # Return a coroutine that can be awaited
        return jina_rerank_async(query, documents, top_n, model)
    except RuntimeError:
        # No running loop, we're in sync context
        # Create a new event loop and run the async function
        return asyncio.run(jina_rerank_async(query, documents, top_n, model))