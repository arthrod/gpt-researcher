import asyncio
<<<<<<< HEAD
from typing import Dict, List
=======
import contextlib
import logging
>>>>>>> newdev

from fastapi import WebSocket

from backend.chat import ChatAgentWithMemory
from backend.report_type import BasicReport, DetailedReport
from backend.server.server_utils import CustomLogsHandler
from gpt_researcher.actions import stream_output  # Import stream_output
from gpt_researcher.utils.enum import ReportType, Tone
from multi_agents.main import run_research_task

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manage websockets"""

    def __init__(self):
        """Initialize the WebSocketManager class."""
        self.active_connections: list[WebSocket] = []
        self.sender_tasks: dict[WebSocket, asyncio.Task] = {}
        self.message_queues: dict[WebSocket, asyncio.Queue] = {}
        self.chat_agent = None

    async def start_sender(self, websocket: WebSocket):
        """
        Run the per-connection sender loop that pulls messages from the connection's outgoing queue and sends them over the WebSocket.
        
        The coroutine reads from self.message_queues[websocket]. Behavior:
        - If no queue exists for the websocket, returns immediately.
        - Messages are pulled until a shutdown sentinel (None) is received or the connection becomes inactive.
        - A message equal to "ping" is replied to with "pong"; all other non-None messages are sent as text.
        - Exceptions encountered while sending cause the loop to exit.
        
        This function has the side effects of sending messages on the provided WebSocket and consuming/acknowledging items from the associated asyncio.Queue.
        """
        queue = self.message_queues.get(websocket)
        if not queue:
            return

        while True:
            try:
                message = await queue.get()
                if message is None:  # Shutdown signal
                    break

                if websocket in self.active_connections:
                    if message == "ping":
                        await websocket.send_text("pong")
                    else:
                        await websocket.send_text(message)
                else:
                    break
            except Exception as e:
                print(f"Error in sender task: {e}")
                break

    async def connect(self, websocket: WebSocket):
        """Connect a websocket."""
        try:
            await websocket.accept()
            self.active_connections.append(websocket)
            self.message_queues[websocket] = asyncio.Queue()
            self.sender_tasks[websocket] = asyncio.create_task(
                self.start_sender(websocket)
            )
        except Exception as e:
            print(f"Error connecting websocket: {e}")
            if websocket in self.active_connections:
                await self.disconnect(websocket)

    async def disconnect(self, websocket: WebSocket):
        """Disconnect a websocket."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            if websocket in self.sender_tasks:
                self.sender_tasks[websocket].cancel()
                await self.message_queues[websocket].put(None)
                del self.sender_tasks[websocket]
            if websocket in self.message_queues:
                del self.message_queues[websocket]
            try:
                with contextlib.suppress(Exception):
                    await websocket.close()
            except Exception as e:
                logger.error(f"Error disconnecting websocket: {e}")

<<<<<<< HEAD
    async def start_streaming(self, task, report_type, report_source, source_urls, document_urls, tone, websocket, headers=None, query_domains=[], mcp_enabled=False, mcp_strategy="fast", mcp_configs=[]):
        """
        Start a research run and stream its output, then initialize a chat agent with the produced report.
        
        Converts the provided `tone` (string) to the Tone enum, sets a default config path, and calls `run_agent` with the given parameters (including optional MCP configuration). After the report is produced, replaces `self.chat_agent` with a new ChatAgentWithMemory initialized from the report, config path, and headers, and returns the report.
        
        Parameters:
            task (str): Research query or task description to run.
            report_type (str): Report flavor identifier (e.g., "basic", "detailed", or "multi_agents").
            report_source (str): Source identifier for the report generation.
            source_urls (list[str]): URLs to fetch source content from.
            document_urls (list[str]): URLs pointing to documents to include in the report.
            tone (str|Tone): Name of a Tone enum member or a Tone instance; converted to Tone internally.
            websocket: WebSocket used for streaming logs/output to the client.
            headers (dict, optional): Optional HTTP headers to pass to downstream components.
            query_domains (list[str], optional): Domains to constrain or prioritize during research.
            mcp_enabled (bool, optional): If True, enables MCP-related behavior in the agent run.
            mcp_strategy (str, optional): MCP strategy name (defaults to "fast").
            mcp_configs (list, optional): Additional MCP configuration entries.
        
        Returns:
            str: Generated report content.
        """
=======
    async def start_streaming(
        self,
        task,
        report_type,
        report_source,
        source_urls,
        document_urls,
        tone,
        websocket,
        headers=None,
        query_domains=None,
        mcp_enabled=False,
        mcp_strategy="fast",
        mcp_configs=None,
    ):
        """Start streaming the output."""
        if mcp_configs is None:
            mcp_configs = []
        if query_domains is None:
            query_domains = []
>>>>>>> newdev
        tone = Tone[tone]
        # add customized JSON config file path here
        config_path = "default"

        # Pass MCP parameters to run_agent
        report = await run_agent(
<<<<<<< HEAD
            task, report_type, report_source, source_urls, document_urls, tone, websocket,
            headers=headers, query_domains=query_domains, config_path=config_path,
            mcp_enabled=mcp_enabled, mcp_strategy=mcp_strategy, mcp_configs=mcp_configs
=======
            task,
            report_type,
            report_source,
            source_urls,
            document_urls,
            tone,
            websocket,
            headers=headers,
            query_domains=query_domains,
            config_path=config_path,
            mcp_enabled=mcp_enabled,
            mcp_strategy=mcp_strategy,
            mcp_configs=mcp_configs,
>>>>>>> newdev
        )

        # Create new Chat Agent whenever a new report is written
        self.chat_agent = ChatAgentWithMemory(report, config_path, headers)
        return report

    async def chat(self, message, websocket):
        """
        Delegate an incoming chat message to the active chat agent or notify the client that no knowledge is available.
        
        If a ChatAgentWithMemory is attached to this manager, forwards the message to its chat(...) coroutine along with the websocket. If no chat agent exists, sends a JSON message over the websocket with type "chat" and content prompting the user to run research first.
        
        Parameters:
            message: The chat message payload to deliver to the agent (content and structure expected by the agent).
        """
        if self.chat_agent:
            await self.chat_agent.chat(message, websocket)
        else:
            await websocket.send_json({
                "type": "chat",
                "content": "Knowledge empty, please run the research first to obtain knowledge",
            })

<<<<<<< HEAD
async def run_agent(task, report_type, report_source, source_urls, document_urls, tone: Tone, websocket, stream_output=stream_output, headers=None, query_domains=[], config_path="", return_researcher=False, mcp_enabled=False, mcp_strategy="fast", mcp_configs=[]):
    """
    Run a research agent to produce a report (Basic, Detailed, or multi-agent) and stream progress to a websocket.
    
    This function coordinates researcher execution and streaming logs:
    - For "multi_agents", it calls run_research_task and extracts the returned "report".
    - For DetailedReport or other types, it constructs and awaits a DetailedReport or BasicReport instance respectively.
    - If MCP (multi-cloud/proxy) is enabled, it updates environment variables ("RETRIEVER" to include "mcp" and "MCP_STRATEGY") and emits an initialization log to the websocket.
    
    Parameters:
        task (str): The research query or task description.
        report_type (str): One of "multi_agents" or the values from ReportType (e.g., DetailedReport.value) determining which researcher to run.
        report_source (str): Identifier of the report source to include in researcher construction.
        source_urls (Sequence[str]): URLs to be used as sources for the report.
        document_urls (Sequence[str]): Document URLs to include in the research.
        tone (Tone): Tone enum controlling stylistic output of the report.
        websocket: WebSocket-like object used by CustomLogsHandler to stream logs and messages.
        stream_output (callable or sentinel): Streaming output handler passed to run_research_task (defaults to module-level stream_output).
        headers (Mapping, optional): Optional HTTP headers or metadata forwarded to researchers.
        query_domains (Sequence[str], optional): Domains to constrain retrieval/querying.
        config_path (str, optional): Configuration path or profile name for report generation.
        return_researcher (bool, optional): If True (and report_type is not "multi_agents"), also return the underlying gpt_researcher object as a second tuple element.
        mcp_enabled (bool, optional): When True and mcp_configs provided, enables MCP-related environment configuration and notifies logs.
        mcp_strategy (str, optional): Strategy name to set in MCP_STRATEGY when MCP is enabled.
        mcp_configs (Sequence, optional): MCP server/config entries; presence enables MCP initialization when mcp_enabled is True.
    
    Returns:
        str or tuple: The produced report string. If return_researcher is True for non-"multi_agents" runs, returns (report, researcher.gpt_researcher).
    
    Side effects:
    - Sends structured log messages to the provided websocket via CustomLogsHandler.
    - May modify environment variables "RETRIEVER" and "MCP_STRATEGY" when MCP is enabled.
    """
=======

async def run_agent(
    task,
    report_type,
    report_source,
    source_urls,
    document_urls,
    tone: Tone,
    websocket,
    stream_output=stream_output,
    headers=None,
    query_domains=None,
    config_path="",
    return_researcher=False,
    mcp_enabled=False,
    mcp_strategy="fast",
    mcp_configs=None,
):
    """Run the agent."""
>>>>>>> newdev
    # Create logs handler for this research task
    if mcp_configs is None:
        mcp_configs = []
    if query_domains is None:
        query_domains = []
    logs_handler = CustomLogsHandler(websocket, task)

    # Set up MCP configuration if enabled
    if mcp_enabled and mcp_configs:
        import os

        current_retriever = os.getenv("RETRIEVER", "tavily")
        if "mcp" not in current_retriever:
            # Add MCP to existing retrievers
            os.environ["RETRIEVER"] = f"{current_retriever},mcp"

        # Set MCP strategy
        os.environ["MCP_STRATEGY"] = mcp_strategy

<<<<<<< HEAD
        print(f"🔧 MCP enabled with strategy '{mcp_strategy}' and {len(mcp_configs)} server(s)")
=======
        print(
            f"🔧 MCP enabled with strategy '{mcp_strategy}' and {len(mcp_configs)} server(s)"
        )
>>>>>>> newdev
        await logs_handler.send_json({
            "type": "logs",
            "content": "mcp_init",
            "output": f"🔧 MCP enabled with strategy '{mcp_strategy}' and {len(mcp_configs)} server(s)",
        })

    # Initialize researcher based on report type
    if report_type == "multi_agents":
        report = await run_research_task(
            query=task,
            websocket=logs_handler,  # Use logs_handler instead of raw websocket
            stream_output=stream_output,
            tone=tone,
<<<<<<< HEAD
            headers=headers
=======
            headers=headers,
>>>>>>> newdev
        )
        report = report.get("report", "")

    elif report_type == ReportType.DetailedReport.value:
        researcher = DetailedReport(
            query=task,
            query_domains=query_domains,
            report_type=report_type,
            report_source=report_source,
            source_urls=source_urls,
            document_urls=document_urls,
            tone=tone,
            config_path=config_path,
            websocket=logs_handler,  # Use logs_handler instead of raw websocket
            headers=headers,
            mcp_configs=mcp_configs if mcp_enabled else None,
            mcp_strategy=mcp_strategy if mcp_enabled else None,
        )
        report = await researcher.run()

    else:
        researcher = BasicReport(
            query=task,
            query_domains=query_domains,
            report_type=report_type,
            report_source=report_source,
            source_urls=source_urls,
            document_urls=document_urls,
            tone=tone,
            config_path=config_path,
            websocket=logs_handler,  # Use logs_handler instead of raw websocket
            headers=headers,
            mcp_configs=mcp_configs if mcp_enabled else None,
            mcp_strategy=mcp_strategy if mcp_enabled else None,
        )
        report = await researcher.run()

    if report_type != "multi_agents" and return_researcher:
        return report, researcher.gpt_researcher
    else:
        return report
