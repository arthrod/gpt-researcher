"""
MCP Client Management Module

Handles MCP client creation, configuration conversion, and connection management.
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    HAS_MCP_ADAPTERS = True
except ImportError:
    HAS_MCP_ADAPTERS = False

logger = logging.getLogger(__name__)


class MCPClientManager:
    """
    Manages MCP client lifecycle and configuration.
    
    Responsible for:
    - Converting GPT Researcher MCP configs to langchain format
    - Creating and managing MultiServerMCPClient instances
    - Handling client cleanup and resource management
    """

    def __init__(self, mcp_configs: List[Dict[str, Any]]):
        """
        Initialize the MCP client manager.
        
        Args:
            mcp_configs: List of MCP server configurations from GPT Researcher
        """
        self.mcp_configs = mcp_configs or []
        self._client = None
        self._client_lock = asyncio.Lock()

    def convert_configs_to_langchain_format(self) -> Dict[str, Dict[str, Any]]:
        """
        Convert the stored GPT Researcher MCP server configurations into the format expected by langchain-mcp-adapters.
        
        This inspects each entry in self.mcp_configs and produces a per-server configuration mapping suitable for MultiServerMCPClient:
        - Derives a server name from config["name"] or uses "mcp_server_{i+1}" when absent.
        - Infers transport from connection_url:
          - ws/wss -> transport "websocket" with "url".
          - http/https -> transport "streamable_http" with "url".
          - Otherwise falls back to config["connection_type"] (default "stdio"); the URL is included only for transports that accept it (e.g., "websocket", "streamable_http", "http").
        - If no connection_url is present, uses config["connection_type"] or "stdio" as transport.
        - For "stdio" transports with a "command":
          - Includes "command".
          - Normalizes "args" into a list (splits a string value) and includes it as "args".
          - Includes "env" when provided.
        - If "connection_token" is present, includes it as "token".
        
        Returns:
            Dict[str, Dict[str, Any]]: Mapping of server names to server configuration dictionaries ready for MultiServerMCPClient.
        """
        server_configs = {}

        for i, config in enumerate(self.mcp_configs):
            # Generate server name
            server_name = config.get("name", f"mcp_server_{i+1}")

            # Build the server config
            server_config = {}

            # Auto-detect transport type from URL if provided
            connection_url = config.get("connection_url")
            if connection_url:
                if connection_url.startswith(("wss://", "ws://")):
                    server_config["transport"] = "websocket"
                    server_config["url"] = connection_url
                elif connection_url.startswith(("https://", "http://")):
                    server_config["transport"] = "streamable_http"
                    server_config["url"] = connection_url
                else:
                    # Fallback to specified connection_type or stdio
                    connection_type = config.get("connection_type", "stdio")
                    server_config["transport"] = connection_type
                    if connection_type in ["websocket", "streamable_http", "http"]:
                        server_config["url"] = connection_url
            else:
                # No URL provided, use stdio (default) or specified connection_type
                connection_type = config.get("connection_type", "stdio")
                server_config["transport"] = connection_type

            # Handle stdio transport configuration
            if server_config.get("transport") == "stdio":
                if config.get("command"):
                    server_config["command"] = config["command"]

                    # Handle server_args
                    server_args = config.get("args", [])
                    if isinstance(server_args, str):
                        server_args = server_args.split()
                    server_config["args"] = server_args

                    # Handle environment variables
                    server_env = config.get("env", {})
                    if server_env:
                        server_config["env"] = server_env

            # Add authentication if provided
            if config.get("connection_token"):
                server_config["token"] = config["connection_token"]

            server_configs[server_name] = server_config

        return server_configs

    async def get_or_create_client(self) -> Optional[object]:
        """
        Get or create a MultiServerMCPClient and cache it on the manager.
        
        If a client already exists, it is returned. If the langchain-mcp-adapters are missing, no MCP configurations are provided, or client creation fails, this method returns None. On success the created client is stored in self._client for subsequent calls.
        
        Returns:
            Optional[object]: The existing or newly created MultiServerMCPClient instance, or None if unavailable or creation failed.
        """
        async with self._client_lock:
            if self._client is not None:
                return self._client

            if not HAS_MCP_ADAPTERS:
                logger.error("langchain-mcp-adapters not installed")
                return None

            if not self.mcp_configs:
                logger.error("No MCP server configurations found")
                return None

            try:
                # Convert configs to langchain format
                server_configs = self.convert_configs_to_langchain_format()
                logger.info(f"Creating MCP client for {len(server_configs)} server(s)")

                # Initialize the MultiServerMCPClient
                self._client = MultiServerMCPClient(server_configs)

                return self._client

            except Exception as e:
                logger.error(f"Error creating MCP client: {e}")
                return None

    async def close_client(self):
        """
        Properly close the MCP client and clean up resources.
        """
        async with self._client_lock:
            if self._client is not None:
                try:
                    # Since MultiServerMCPClient doesn't support context manager
                    # or explicit close methods in langchain-mcp-adapters 0.1.0,
                    # we just clear the reference and let garbage collection handle it
                    logger.debug("Releasing MCP client reference")
                except Exception as e:
                    logger.error(f"Error during MCP client cleanup: {e}")
                finally:
                    # Always clear the reference
                    self._client = None

    async def get_all_tools(self) -> List:
        """
        Return a list of all tools exposed by the configured MCP servers.
        
        This coroutine obtains (or lazily creates) the MCP client and requests tools from all configured servers.
        If no client can be created, no tools are available, or an error occurs while fetching tools, an empty list
        is returned rather than raising.
        
        Returns:
            List: A list of tool objects provided by the MCP servers; empty if none are available or on error.
        """
        client = await self.get_or_create_client()
        if not client:
            return []

        try:
            # Get tools from all servers
            all_tools = await client.get_tools()

            if all_tools:
                logger.info(f"Loaded {len(all_tools)} total tools from MCP servers")
                return all_tools
            else:
                logger.warning("No tools available from MCP servers")
                return []

        except Exception as e:
            logger.error(f"Error getting MCP tools: {e}")
            return []