"""
MCP Tool Selection Module

Handles intelligent tool selection using LLM analysis.
"""
<<<<<<< HEAD

import json
import logging
=======
import json
import logging
from typing import List
>>>>>>> 1027e1d0 (Fix linting issues)

logger = logging.getLogger(__name__)


class MCPToolSelector:
    """
    Handles intelligent selection of MCP tools using LLM analysis.

    Responsible for:
    - Analyzing available tools with LLM
    - Selecting the most relevant tools for a query
    - Providing fallback selection mechanisms
    """

    def __init__(self, cfg, researcher=None):
        """
        Initialize the tool selector.

        Args:
            cfg: Configuration object with LLM settings
            researcher: Researcher instance for cost tracking
        """
        self.cfg = cfg
        self.researcher = researcher

    async def select_relevant_tools(
        self, query: str, all_tools: list, max_tools: int = 3
    ) -> list:
        """
<<<<<<< HEAD
        Use LLM to select the most relevant tools for the research query.

        Args:
            query: Research query
            all_tools: List of all available tools
            max_tools: Maximum number of tools to select (default: 3)

=======
        Select the most relevant tools for a research query using an LLM, with a pattern-based fallback.
        
        Uses the configured strategic LLM to rank and pick up to `max_tools` from `all_tools`. Each element of
        `all_tools` is expected to expose `name` and `description` attributes; the function sends indexed tool
        metadata to the LLM and expects a JSON response containing a `selected_tools` list (each entry should
        include an `index` referencing `all_tools`, and may include `reason` and `relevance_score`). If the
        LLM returns no usable response or returned JSON cannot be parsed, the method falls back to a
        deterministic pattern-based selection.
        
        Parameters:
            query: The research query driving tool selection.
            all_tools: List of available tool objects (each should provide `.name` and `.description`).
            max_tools: Maximum number of tools to return; will be capped to len(all_tools).
        
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        Returns:
            List of selected tool objects (a subset of `all_tools`). If LLM selection fails or yields no
            selections, a fallback list determined by heuristic scoring is returned.
        """
        if not all_tools:
            return []

        if len(all_tools) < max_tools:
            max_tools = len(all_tools)

<<<<<<< HEAD
        logger.info(
            f"Using LLM to select {max_tools} most relevant tools from {len(all_tools)} available"
        )
=======
        logger.info(f"Using LLM to select {max_tools} most relevant tools from {len(all_tools)} available")
>>>>>>> 1027e1d0 (Fix linting issues)

        # Create tool descriptions for LLM analysis
        tools_info = []
        for i, tool in enumerate(all_tools):
            tool_info = {
                "index": i,
                "name": tool.name,
                "description": tool.description or "No description available",
            }
            tools_info.append(tool_info)

        # Import here to avoid circular imports
        from ..prompts import PromptFamily

        # Create prompt for intelligent tool selection
        prompt = PromptFamily.generate_mcp_tool_selection_prompt(
            query, tools_info, max_tools
        )

        try:
            # Call LLM for tool selection
            response = await self._call_llm_for_tool_selection(prompt)

            if not response:
                logger.warning("No LLM response for tool selection, using fallback")
                return self._fallback_tool_selection(all_tools, max_tools)

            # Log a preview of the LLM response for debugging
            response_preview = (
                response[:500] + "..." if len(response) > 500 else response
            )
            logger.debug(f"LLM tool selection response: {response_preview}")

            # Parse LLM response
            try:
                selection_result = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re

                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    try:
                        selection_result = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        logger.warning("Could not parse extracted JSON, using fallback")
                        return self._fallback_tool_selection(all_tools, max_tools)
                else:
                    logger.warning("No JSON found in LLM response, using fallback")
                    return self._fallback_tool_selection(all_tools, max_tools)

            selected_tools = []

            # Process selected tools
            for tool_selection in selection_result.get("selected_tools", []):
                tool_index = tool_selection.get("index")
                tool_name = tool_selection.get("name", "")
                reason = tool_selection.get("reason", "")
                relevance_score = tool_selection.get("relevance_score", 0)

                if tool_index is not None and 0 <= tool_index < len(all_tools):
                    selected_tools.append(all_tools[tool_index])
<<<<<<< HEAD
                    logger.info(
                        f"Selected tool '{tool_name}' (score: {relevance_score}): {reason}"
                    )
=======
                    logger.info(f"Selected tool '{tool_name}' (score: {relevance_score}): {reason}")
>>>>>>> 1027e1d0 (Fix linting issues)

            if len(selected_tools) == 0:
                logger.warning("No tools selected by LLM, using fallback selection")
                return self._fallback_tool_selection(all_tools, max_tools)

            # Log the overall selection reasoning
            selection_reasoning = selection_result.get(
                "selection_reasoning", "No reasoning provided"
            )
            logger.info(f"LLM selection strategy: {selection_reasoning}")

            logger.info(f"LLM selected {len(selected_tools)} tools for research")
            return selected_tools

        except Exception as e:
            logger.error(f"Error in LLM tool selection: {e}")
            logger.warning("Falling back to pattern-based selection")
            return self._fallback_tool_selection(all_tools, max_tools)

    async def _call_llm_for_tool_selection(self, prompt: str) -> str:
        """
<<<<<<< HEAD
        Call the LLM using the existing create_chat_completion function for tool selection.

        Args:
            prompt (str): The prompt to send to the LLM.

=======
        Call the configured LLM to generate a tool-selection response for the given prompt.
        
        This sends the prompt as a single user message to the strategic LLM model configured on self.cfg using a deterministic temperature (0.0). If a researcher with an add_costs method is attached, that method will be used as the cost_callback for the LLM call. On failure (including missing configuration or any runtime error) the method returns an empty string.
        
        Parameters:
            prompt (str): The prompt text sent to the LLM as the user message.
        
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        Returns:
            str: The LLM's text response, or an empty string on error or if no configuration is available.
        """
        if not self.cfg:
            logger.warning("No config available for LLM call")
            return ""

        try:
            from ..utils.llm import create_chat_completion

            # Create messages for the LLM
            messages = [{"role": "user", "content": prompt}]

            # Use the strategic LLM for tool selection (as it's more complex reasoning)
            result = await create_chat_completion(
                model=self.cfg.strategic_llm_model,
                messages=messages,
                temperature=0.0,  # Low temperature for consistent tool selection
                llm_provider=self.cfg.strategic_llm_provider,
                llm_kwargs=self.cfg.llm_kwargs,
                cost_callback=self.researcher.add_costs
                if self.researcher and hasattr(self.researcher, "add_costs")
                else None,
            )
            return result
        except Exception as e:
            logger.error(f"Error calling LLM for tool selection: {e}")
            return ""

    def _fallback_tool_selection(self, all_tools: list, max_tools: int) -> list:
        """
<<<<<<< HEAD
        Fallback tool selection using pattern matching if LLM selection fails.

        Args:
            all_tools: List of all available tools
            max_tools: Maximum number of tools to select

=======
        Select up to max_tools using a simple pattern-based heuristic when LLM selection fails.
        
        This case-insensitive fallback scans each tool's name and description for research-oriented keywords
        (e.g., "search", "fetch", "find", "query", "browse", "describe"). Matches in the tool name are
        counted as higher relevance than matches in the description (name match weight = 3, description
        match weight = 1). Tools with a total score of zero are ignored. Results are sorted by score
        (descending) and the top max_tools are returned. If no tools match, an empty list is returned.
        
        Parameters:
            all_tools (List): Iterable of tool objects; each must expose .name (str) and .description (optional str).
            max_tools (int): Maximum number of tools to return.
        
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        Returns:
            List: Selected tool objects (up to max_tools), ordered by descending relevance score.
        """
        # Define patterns for research-relevant tools
        research_patterns = [
<<<<<<< HEAD
            "search",
            "get",
            "read",
            "fetch",
            "find",
            "list",
            "query",
            "lookup",
            "retrieve",
            "browse",
            "view",
            "show",
            "describe",
=======
            'search', 'get', 'read', 'fetch', 'find', 'list', 'query',
            'lookup', 'retrieve', 'browse', 'view', 'show', 'describe'
>>>>>>> 1027e1d0 (Fix linting issues)
        ]

        scored_tools = []

        for tool in all_tools:
            tool_name = tool.name.lower()
            tool_description = (tool.description or "").lower()

            # Calculate relevance score based on pattern matching
            score = 0
            for pattern in research_patterns:
                if pattern in tool_name:
                    score += 3
                if pattern in tool_description:
                    score += 1

            if score > 0:
                scored_tools.append((tool, score))

        # Sort by score and take top tools
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        selected_tools = [tool for tool, score in scored_tools[:max_tools]]

        for i, (tool, score) in enumerate(scored_tools[:max_tools]):
<<<<<<< HEAD
            logger.info(f"Fallback selected tool {i + 1}: {tool.name} (score: {score})")

        return selected_tools
=======
            logger.info(f"Fallback selected tool {i+1}: {tool.name} (score: {score})")

        return selected_tools
>>>>>>> 1027e1d0 (Fix linting issues)
