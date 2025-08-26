import json
import re

import json_repair

from gpt_researcher.prompts import PromptFamily
from gpt_researcher.utils.llm import create_chat_completion


async def choose_agent(
    query,
    cfg,
    parent_query=None,
    cost_callback: callable | None = None,
    headers=None,
    prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
    **kwargs,
):
    """
    Automatically selects an agent for a research task by querying the configured LLM.
    
    If parent_query is provided, it is prepended to query ("parent_query - query") to give context. The function sends the combined task to the LLM (using prompt_family.auto_agent_instructions()) and expects a JSON response containing the keys "server" and "agent_role_prompt". Returns a tuple (agent_name, agent_role_prompt). If the LLM response cannot be parsed as JSON, the function delegates recovery to handle_json_error and returns its result.
    """
    query = f"{parent_query} - {query}" if parent_query else f"{query}"
    response = None  # Initialize response to ensure it's defined

    try:
        response = await create_chat_completion(
            model=cfg.smart_llm_model,
            messages=[
                {
                    "role": "system",
                    "content": f"{prompt_family.auto_agent_instructions()}",
                },
                {"role": "user", "content": f"task: {query}"},
            ],
            temperature=0.15,
            llm_provider=cfg.smart_llm_provider,
            llm_kwargs=cfg.llm_kwargs,
            cost_callback=cost_callback,
            **kwargs,
        )

        agent_dict = json.loads(response)
        return agent_dict["server"], agent_dict["agent_role_prompt"]

<<<<<<< HEAD
    except Exception as e:
        return await handle_json_error(response, e)
=======
    except Exception:
        return await handle_json_error(response)
>>>>>>> 1027e1d0 (Fix linting issues)


async def handle_json_error(response):
    try:
        agent_dict = json_repair.loads(response)
        if agent_dict.get("server") and agent_dict.get("agent_role_prompt"):
            return agent_dict["server"], agent_dict["agent_role_prompt"]
    except Exception as e:
        print(f"⚠️ Error in reading JSON and failed to repair with json_repair: {e}")
        print(f"⚠️ LLM Response: `{response}`")

    json_string = extract_json_with_regex(response)
    if json_string:
        try:
            json_data = json.loads(json_string)
            return json_data["server"], json_data["agent_role_prompt"]
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

    print("No JSON found in the string. Falling back to Default Agent.")
    return "Default Agent", (
        "You are an AI critical thinker research assistant. Your sole purpose is to write well written, "
        "critically acclaimed, objective and structured reports on given text."
    )


def extract_json_with_regex(response):
    json_match = re.search(r"{.*?}", response, re.DOTALL)
    if json_match:
        return json_match.group(0)
    return None
