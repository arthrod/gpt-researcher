"""
MCP Streaming Utilities Module

Handles websocket streaming and logging for MCP operations.
"""

import asyncio
import logging
<<<<<<< HEAD
=======

>>>>>>> newdev
from typing import Any

logger = logging.getLogger(__name__)


class MCPStreamer:
    """
    Handles streaming output for MCP operations.

    Responsible for:
    - Streaming logs to websocket
    - Synchronous/asynchronous logging
    - Error handling in streaming
    """

    def __init__(self, websocket=None):
        """
        Initialize the MCP streamer.

        Args:
            websocket: WebSocket for streaming output
        """
        self.websocket = websocket

    async def stream_log(self, message: str, data: Any = None):
<<<<<<< HEAD
        """
        Log `message` locally and, if a websocket is configured, attempt to stream it as an MCP "logs" message.
        
        The optional `data` value is forwarded as the streaming `metadata`. Streaming errors are caught and logged locally; they are not propagated to callers.
        """
=======
        """Stream a log message to the websocket if available."""
>>>>>>> newdev
        logger.info(message)

        if self.websocket:
            try:
                from ..actions.utils import stream_output

                await stream_output(
                    type="logs",
                    content="mcp_retriever",
                    output=message,
                    websocket=self.websocket,
                    metadata=data,
                )
            except Exception as e:
                logger.error(f"Error streaming log: {e}")

    def stream_log_sync(self, message: str, data: Any = None):
<<<<<<< HEAD
        """
        Synchronous wrapper around stream_log that logs locally and forwards the message to the websocket-aware async streamer.
        
        If an asyncio event loop is running, this schedules the async stream_log as a task; otherwise it runs stream_log to completion. All streaming errors are caught and logged (they are not propagated).
        
        Parameters:
            message (str): Log message to emit.
            data (Any, optional): Optional metadata sent alongside the log to the websocket (e.g., context or structured payload).
        """
=======
        """Synchronous version of stream_log for use in sync contexts."""
>>>>>>> newdev
        logger.info(message)

        if self.websocket:
            try:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        task = asyncio.create_task(self.stream_log(message, data))
                        # Store task reference to prevent garbage collection
                        # The task will be cleaned up when it completes
                        if not hasattr(self, "_background_tasks"):
                            self._background_tasks = set()
                        self._background_tasks.add(task)
                        task.add_done_callback(self._background_tasks.discard)
                    else:
                        loop.run_until_complete(self.stream_log(message, data))
                except RuntimeError:
                    logger.debug("Could not stream log: no running event loop")
            except Exception as e:
                logger.error(f"Error in sync log streaming: {e}")

    async def stream_stage_start(self, stage: str, description: str):
        """Stream the start of a research stage."""
        await self.stream_log(f"🔧 {stage}: {description}")

    async def stream_stage_complete(self, stage: str, result_count: int | None = None):
        """Stream the completion of a research stage."""
        if result_count is not None:
            await self.stream_log(f"✅ {stage} completed: {result_count} results")
        else:
            await self.stream_log(f"✅ {stage} completed")

    async def stream_tool_selection(self, selected_count: int, total_count: int):
        """Stream tool selection information."""
        await self.stream_log(
            f"🧠 Using LLM to select {selected_count} most relevant tools from {total_count} available"
        )

    async def stream_tool_execution(self, tool_name: str, step: int, total: int):
        """Stream tool execution progress."""
        await self.stream_log(f"🔍 Executing tool {step}/{total}: {tool_name}")

    async def stream_research_results(
        self, result_count: int, total_chars: int | None = None
    ):
        """Stream research results summary."""
        if total_chars:
            await self.stream_log(
                f"✅ MCP research completed: {result_count} results obtained ({total_chars:,} chars)"
            )
        else:
            await self.stream_log(
                f"✅ MCP research completed: {result_count} results obtained"
            )

    async def stream_error(self, error_msg: str):
        """Stream error messages."""
        await self.stream_log(f"❌ {error_msg}")

    async def stream_warning(self, warning_msg: str):
        """Stream warning messages."""
        await self.stream_log(f"⚠️ {warning_msg}")

    async def stream_info(self, info_msg: str):
<<<<<<< HEAD
        """
        Stream an informational message to the configured output.
        
        The message will be sent with an informational prefix ("ℹ️ ") and forwarded to the stream_log mechanism so it is logged locally and (if configured) streamed over the websocket.
        
        Parameters:
            info_msg (str): The informational text to send; it will be prefixed with "ℹ️ ".
        """
        await self.stream_log(f"ℹ️ {info_msg}")
=======
        """Stream informational messages."""
        await self.stream_log(f"i {info_msg}")
>>>>>>> newdev
