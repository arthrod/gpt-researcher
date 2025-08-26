import os
import sys
import uuid

from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
import json

from gpt_researcher.utils.enum import Tone
from multi_agents.agents import ChiefEditorAgent

# Run with LangSmith if API key is set
if os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
load_dotenv()


def open_task():
    # Get the directory of the current script
    """
    Load and return the task configuration from task.json, optionally overriding the model via the STRATEGIC_LLM environment variable.
    
    Reads task.json from the same directory as this module and returns its contents as a dict. If the file is empty or evaluates to false, raises an Exception indicating a valid task.json is required. If the environment variable STRATEGIC_LLM is set and contains a colon, the substring after the first colon is used to override task["model"].
    
    Returns:
        dict: The parsed task configuration.
    
    Raises:
        Exception: If the loaded task is empty/falsey.
    
    Notes:
        - The function will propagate file I/O and JSON parsing errors (e.g., FileNotFoundError, JSONDecodeError) raised by open() and json.load().
        - Current implementation has a bug: if STRATEGIC_LLM is set but does not contain a colon, it attempts to assign an undefined variable and will raise a NameError.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path to task.json
<<<<<<< HEAD
    task_json_path = os.path.join(current_dir, "task.json")

    with open(task_json_path) as f:
=======
    task_json_path = os.path.join(current_dir, 'task.json')

    with open(task_json_path, 'r') as f:
>>>>>>> 1027e1d0 (Fix linting issues)
        task = json.load(f)

    if not task:
        raise Exception(
            "No task found. Please ensure a valid task.json file is present in the multi_agents directory and contains the necessary task information."
        )

    # Override model with STRATEGIC_LLM if defined in environment
    strategic_llm = os.environ.get("STRATEGIC_LLM")
    if strategic_llm and ":" in strategic_llm:
        # Extract the model name (part after the first colon)
        model_name = strategic_llm.split(":", 1)[1]
        task["model"] = model_name
    elif strategic_llm:
        task["model"] = model_name

    return task


async def run_research_task(
    query, websocket=None, stream_output=None, tone=Tone.Objective, headers=None
):
    task = open_task()
    task["query"] = query

    chief_editor = ChiefEditorAgent(task, websocket, stream_output, tone, headers)
    research_report = await chief_editor.run_research_task()

    if websocket and stream_output:
        await stream_output("logs", "research_report", research_report, websocket)

    return research_report


async def main():
    task = open_task()

    chief_editor = ChiefEditorAgent(task)
    research_report = await chief_editor.run_research_task(task_id=uuid.uuid4())

    return research_report


if __name__ == "__main__":
    asyncio.run(main())
