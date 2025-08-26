import asyncio
import json
import logging
import os
import re
import shutil
import time
import traceback

from collections.abc import Awaitable
from datetime import datetime
from typing import Any

from fastapi.responses import JSONResponse

from backend.utils import write_md_to_pdf, write_md_to_word, write_text_to_md
from gpt_researcher import GPTResearcher
from gpt_researcher.actions import stream_output
from gpt_researcher.document.document import DocumentLoader
from multi_agents.main import run_research_task

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class CustomLogsHandler:
    """Custom handler to capture streaming logs from the research process"""

    def __init__(self, websocket, task: str):
        self.logs = []
        self.websocket = websocket
        sanitized_filename = sanitize_filename(f"task_{int(time.time())}_{task}")
        self.log_file = os.path.join("outputs", f"{sanitized_filename}.json")
        self.timestamp = datetime.now().isoformat()
        # Initialize log file with metadata
        os.makedirs("outputs", exist_ok=True)
        with open(self.log_file, "w") as f:
            json.dump(
                {
                    "timestamp": self.timestamp,
                    "events": [],
                    "content": {
                        "query": "",
                        "sources": [],
                        "context": [],
                        "report": "",
                        "costs": 0.0,
                    },
                },
                f,
                indent=2,
            )

    async def send_json(self, data: dict[str, Any]) -> None:
        """Store log data and send to websocket"""
        # Send to websocket for real-time display
        if self.websocket:
            await self.websocket.send_json(data)

        # Read current log file
        with open(self.log_file) as f:
            log_data = json.load(f)

        # Update appropriate section based on data type
        if data.get("type") == "logs":
            log_data["events"].append({
                "timestamp": datetime.now().isoformat(),
                "type": "event",
                "data": data,
            })
        else:
            # Update content section for other types of data
            log_data["content"].update(data)

        # Save updated log file
        with open(self.log_file, "w") as f:
            json.dump(log_data, f, indent=2)
        logger.debug(f"Log entry written to: {self.log_file}")


class Researcher:
    def __init__(self, query: str, report_type: str = "research_report"):
        self.query = query
        self.report_type = report_type
        # Generate unique ID for this research task
        self.research_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(query)}"
        # Initialize logs handler with research ID
        self.logs_handler = CustomLogsHandler(None, self.research_id)
        self.researcher = GPTResearcher(
            query=query, report_type=report_type, websocket=self.logs_handler
        )

    async def research(self) -> dict:
        """Conduct research and return paths to generated files"""
        await self.researcher.conduct_research()
        report = await self.researcher.write_report()

        # Generate the files
        sanitized_filename = sanitize_filename(f"task_{int(time.time())}_{self.query}")
        file_paths = await generate_report_files(report, sanitized_filename)

        # Get the JSON log path that was created by CustomLogsHandler
        json_relative_path = os.path.relpath(self.logs_handler.log_file)

        return {
            "output": {
                **file_paths,  # Include PDF, DOCX, and MD paths
                "json": json_relative_path,
            }
        }


def sanitize_filename(filename: str) -> str:
    # Split into components
    prefix, timestamp, *task_parts = filename.split("_")
    task = "_".join(task_parts)

    # Calculate max length for task portion
    # 255 - len(os.getcwd()) - len("\\gpt-researcher\\outputs\\") - len("task_") - len(timestamp) - len("_.json") - safety_margin
    max_task_length = (
        255 - len(os.getcwd()) - 24 - 5 - 10 - 6 - 5
    )  # ~189 chars for task

    # Truncate task if needed (by bytes)
    truncated_task = ""
    byte_count = 0
    for char in task:
        char_bytes = len(char.encode("utf-8"))
        if byte_count + char_bytes <= max_task_length:
            truncated_task += char
            byte_count += char_bytes
        else:
            break

    # Reassemble and clean the filename
    sanitized = f"{prefix}_{timestamp}_{truncated_task}"
    return re.sub(r"[^\w-]", "", sanitized).strip()


async def handle_start_command(websocket, data: str, manager):
    json_data = json.loads(data[6:])
    (
        task,
        report_type,
        source_urls,
        document_urls,
        tone,
        headers,
        report_source,
        query_domains,
        mcp_enabled,
        mcp_strategy,
        mcp_configs,
    ) = extract_command_data(json_data)

    if not task or not report_type:
        print("❌ Error: Missing task or report_type")
        await websocket.send_json({
            "type": "logs",
            "content": "error",
            "output": f"Missing required parameters - task: {task}, report_type: {report_type}",
        })
        return

    # Create logs handler with websocket and task
    logs_handler = CustomLogsHandler(websocket, task)
    # Initialize log content with query
    await logs_handler.send_json({
        "query": task,
        "sources": [],
        "context": [],
        "report": "",
    })

    sanitized_filename = sanitize_filename(f"task_{int(time.time())}_{task}")

    report = await manager.start_streaming(
        task,
        report_type,
        report_source,
        source_urls,
        document_urls,
        tone,
        websocket,
        headers,
        query_domains,
        mcp_enabled,
        mcp_strategy,
        mcp_configs,
    )
    report = str(report)
    file_paths = await generate_report_files(report, sanitized_filename)
    # Add JSON log path to file_paths
    file_paths["json"] = os.path.relpath(logs_handler.log_file)
    await send_file_paths(websocket, file_paths)


async def handle_human_feedback(data: str):
    """Handle human feedback and forward it to the appropriate agent or update research state.

    This function processes human feedback received via WebSocket and forwards it to:
    - Multi-agent workflows that are waiting for human input
    - Research state updates for plan revisions
    - Active research sessions that need guidance

    Args:
        data (str): JSON string containing feedback data with "human_feedback" prefix

    The feedback format expected:
    {
        "type": "human_feedback",
        "content": "user feedback text or null",
        "task_id": "optional task identifier",
        "context": "optional context about what feedback is for"
    }
    """
    try:
        # Remove "human_feedback" prefix and parse JSON
        feedback_data = json.loads(data[14:])
        print(f"Received human feedback: {feedback_data}")

        feedback_content = feedback_data.get("content")
        task_id = feedback_data.get("task_id")
        context = feedback_data.get("context", "general")

        # Log feedback for debugging and audit trail
        logger.info(
            f"Processing human feedback - Task ID: {task_id}, Context: {context}, Content: {feedback_content}"
        )

        # Handle different types of feedback contexts
        if context == "plan_review":
            # This feedback is for research plan review in multi-agent workflows
            await _handle_plan_feedback(feedback_content, task_id)
        elif context == "research_guidance":
            # This feedback provides guidance for ongoing research
            await _handle_research_guidance(feedback_content, task_id)
        elif context == "quality_review":
            # This feedback is for quality review and revision cycles
            await _handle_quality_feedback(feedback_content, task_id)
        else:
            # Generic feedback handling - store for later retrieval
            await _handle_generic_feedback(feedback_content, task_id)

        logger.info(f"Successfully processed human feedback for task: {task_id}")

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse human feedback JSON: {e}")
        print(f"Error: Invalid JSON in human feedback: {e}")
    except Exception as e:
        logger.error(f"Error processing human feedback: {e}")
        print(f"Error processing human feedback: {e}")


async def _handle_plan_feedback(
    feedback_content: str | None, task_id: str | None
) -> None:
    """Handle feedback specifically for research plan review.

    This is used in multi-agent workflows where the human reviews
    the research plan and provides guidance for revision.

    Args:
        feedback_content: The feedback content from the user
        task_id: The task identifier
    """
    # In a production system, this would update the research state
    # and signal the workflow to continue with the feedback
    print(f"Plan feedback for task {task_id}: {feedback_content}")
    # TODO: Implement workflow state update when LangGraph integration is available


async def _handle_research_guidance(
    feedback_content: str | None, task_id: str | None
) -> None:
    """Handle feedback that provides guidance for ongoing research.

    Args:
        feedback_content: The feedback content from the user
        task_id: The task identifier
    """
    print(f"Research guidance for task {task_id}: {feedback_content}")
    # This could influence search parameters, focus areas, or depth of research


async def _handle_quality_feedback(
    feedback_content: str | None, task_id: str | None
) -> None:
    """Handle feedback for quality review and revision cycles.

    Args:
        feedback_content: The feedback content from the user
        task_id: The task identifier
    """
    print(f"Quality feedback for task {task_id}: {feedback_content}")
    # This would be used in reviewer-reviser agent cycles


async def _handle_generic_feedback(
    feedback_content: str | None, task_id: str | None
) -> None:
    """Handle generic feedback that doesn't fit other categories.

    Args:
        feedback_content: The feedback content from the user
        task_id: The task identifier
    """
    print(f"Generic feedback for task {task_id}: {feedback_content}")
    # Store feedback in a way that it can be retrieved by active workflows


async def handle_chat(websocket, data: str, manager) -> None:
    """Handle chat messages from the WebSocket connection.

    Args:
        websocket: The WebSocket connection
        data: The chat data as a JSON string
        manager: The WebSocket manager instance
    """
    json_data = json.loads(data[4:])
    print(f"Received chat message: {json_data.get('message')}")
    await manager.chat(json_data.get("message"), websocket)


async def generate_report_files(report: str, filename: str) -> dict[str, str]:
    pdf_path = await write_md_to_pdf(report, filename)
    docx_path = await write_md_to_word(report, filename)
    md_path = await write_text_to_md(report, filename)
    return {"pdf": pdf_path, "docx": docx_path, "md": md_path}


async def send_file_paths(websocket, file_paths: dict[str, str]):
    await websocket.send_json({"type": "path", "output": file_paths})


def get_config_dict(
    langchain_api_key: str,
    openai_api_key: str,
    tavily_api_key: str,
    google_api_key: str,
    google_cx_key: str,
    bing_api_key: str,
    searchapi_api_key: str,
    serpapi_api_key: str,
    serper_api_key: str,
    searx_url: str,
    jina_api_key: str,
) -> dict[str, str]:
    return {
        "LANGCHAIN_API_KEY": langchain_api_key or os.getenv("LANGCHAIN_API_KEY", ""),
        "OPENAI_API_KEY": openai_api_key or os.getenv("OPENAI_API_KEY", ""),
        "TAVILY_API_KEY": tavily_api_key or os.getenv("TAVILY_API_KEY", ""),
        "GOOGLE_API_KEY": google_api_key or os.getenv("GOOGLE_API_KEY", ""),
        "GOOGLE_CX_KEY": google_cx_key or os.getenv("GOOGLE_CX_KEY", ""),
        "BING_API_KEY": bing_api_key or os.getenv("BING_API_KEY", ""),
        "SEARCHAPI_API_KEY": searchapi_api_key or os.getenv("SEARCHAPI_API_KEY", ""),
        "SERPAPI_API_KEY": serpapi_api_key or os.getenv("SERPAPI_API_KEY", ""),
        "SERPER_API_KEY": serper_api_key or os.getenv("SERPER_API_KEY", ""),
        "JINA_API_KEY": jina_api_key or os.getenv("JINA_API_KEY", ""),
        "SEARX_URL": searx_url or os.getenv("SEARX_URL", ""),
        "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2", "true"),
        "DOC_PATH": os.getenv("DOC_PATH", "./my-docs"),
        "RETRIEVER": os.getenv("RETRIEVER", ""),
        "EMBEDDING_MODEL": os.getenv("OPENAI_EMBEDDING_MODEL", ""),
    }


def update_environment_variables(config: dict[str, str]) -> None:
    """Update environment variables with the provided configuration.

    Args:
        config: Dictionary mapping environment variable names to values
    """
    for key, value in config.items():
        os.environ[key] = value


async def handle_file_upload(file, DOC_PATH: str) -> dict[str, str]:
    file_path = os.path.join(DOC_PATH, os.path.basename(file.filename))
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f"File uploaded to {file_path}")

    document_loader = DocumentLoader(DOC_PATH)
    await document_loader.load()

    return {"filename": file.filename, "path": file_path}


async def handle_file_deletion(filename: str, DOC_PATH: str) -> JSONResponse:
    file_path = os.path.join(DOC_PATH, os.path.basename(filename))
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File deleted: {file_path}")
        return JSONResponse(content={"message": "File deleted successfully"})
    else:
        print(f"File not found: {file_path}")
        return JSONResponse(status_code=404, content={"message": "File not found"})


async def execute_multi_agents(manager) -> Any:
    websocket = manager.active_connections[0] if manager.active_connections else None
    if websocket:
        report = await run_research_task(
            "Is AI in a hype cycle?", websocket, stream_output
        )
        return {"report": report}
    else:
        return JSONResponse(
            status_code=400, content={"message": "No active WebSocket connection"}
        )


async def handle_websocket_communication(websocket, manager):
    running_task: asyncio.Task | None = None

    def run_long_running_task(awaitable: Awaitable) -> asyncio.Task:
        async def safe_run():
            try:
                await awaitable
            except asyncio.CancelledError:
                logger.info("Task cancelled.")
                raise
            except Exception as e:
                logger.error(f"Error running task: {e}\n{traceback.format_exc()}")
                await websocket.send_json({
                    "type": "logs",
                    "content": "error",
                    "output": f"Error: {e}",
                })

        return asyncio.create_task(safe_run())

    try:
        while True:
            try:
                data = await websocket.receive_text()

                if data == "ping":
                    await websocket.send_text("pong")
                elif running_task and not running_task.done():
                    # discard any new request if a task is already running
                    logger.warning(
                        f"Received request while task is already running. Request data preview: {data[: min(20, len(data))]}..."
                    )
                    websocket.send_json({
                        "types": "logs",
                        "output": "Task already running. Please wait.",
                    })
                elif data.startswith("start"):
                    running_task = run_long_running_task(
                        handle_start_command(websocket, data, manager)
                    )
                elif data.startswith("human_feedback"):
                    running_task = run_long_running_task(handle_human_feedback(data))
                elif data.startswith("chat"):
                    running_task = run_long_running_task(
                        handle_chat(websocket, data, manager)
                    )
                else:
                    print("Error: Unknown command or not enough parameters provided.")
            except Exception as e:
                print(f"WebSocket error: {e}")
                break
    finally:
        if running_task and not running_task.done():
            running_task.cancel()


def extract_command_data(json_data: dict[str, Any]) -> tuple[Any, ...]:
    """Extract command data from JSON payload.

    Args:
        json_data: Dictionary containing command parameters

    Returns:
        Tuple containing extracted command parameters in order:
        (task, report_type, source_urls, document_urls, tone, headers,
         report_source, query_domains, mcp_enabled, mcp_strategy, mcp_configs)
    """
    return (
        json_data.get("task"),
        json_data.get("report_type"),
        json_data.get("source_urls"),
        json_data.get("document_urls"),
        json_data.get("tone"),
        json_data.get("headers", {}),
        json_data.get("report_source"),
        json_data.get("query_domains", []),
        json_data.get("mcp_enabled", False),
        json_data.get("mcp_strategy", "fast"),
        json_data.get("mcp_configs", []),
    )
