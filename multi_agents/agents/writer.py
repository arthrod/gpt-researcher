from datetime import datetime
import json5 as json
from .utils.views import print_agent_output
from .utils.llms import call_model

sample_json = """
{
  "table_of_contents": A table of contents in markdown syntax (using '-') based on the research headers and subheaders,
  "introduction": An indepth introduction to the topic in markdown syntax and hyperlink references to relevant sources,
  "conclusion": A conclusion to the entire research based on all research data in markdown syntax and hyperlink references to relevant sources,
  "sources": A list with strings of all used source links in the entire research data in markdown syntax and apa citation format. For example: ['-  Title, year, Author [source url](source)', ...]
}
"""


class WriterAgent:
    def __init__(self, websocket=None, stream_output=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers

    def get_headers(self, research_state: dict):
        return {
            "title": research_state.get("title"),
            "date": "Date",
            "introduction": "Introduction",
            "table_of_contents": "Table of Contents",
            "conclusion": "Conclusion",
            "references": "References",
        }

    async def write_sections(self, research_state: dict):
        """
        Generate an introduction and conclusion for a research report from the provided research state by prompting the configured model and returning the model's JSON response.
        
        The function extracts these keys from `research_state`:
        - `title` (str): Topic/query to write about.
        - `research_data` (any): Research findings and information to be incorporated into the introduction and conclusion.
        - `task` (dict): Task configuration containing at minimum:
          - `model` (str): Model identifier passed to the underlying `call_model`.
          - `follow_guidelines` (bool): If true, the `guidelines` value will be included in the prompt.
          - `guidelines` (str): Optional instructions to be followed by the model.
        
        The model is instructed to return a JSON object matching the file's `sample_json` schema (fields such as `table_of_contents`, `introduction`, `conclusion`, `sources`), and this function returns that parsed JSON-like response.
        
        Parameters:
            research_state (dict): State object described above.
        
        Returns:
            dict: Parsed JSON response from the model containing the generated sections (e.g., `introduction`, `conclusion`, `sources`, `table_of_contents`).
        """
        query = research_state.get("title")
        data = research_state.get("research_data")
        task = research_state.get("task")
        follow_guidelines = task.get("follow_guidelines")
        guidelines = task.get("guidelines")

        prompt = [
            {
                "role": "system",
                "content": "You are a research writer. Your sole purpose is to write a well-written "
                "research reports about a "
                "topic based on research findings and information.\n ",
            },
            {
                "role": "user",
                "content": f"Today's date is {datetime.now().strftime('%d/%m/%Y')}\n."
                f"Query or Topic: {query}\n"
                f"Research data: {data!s}\n"
                f"Your task is to write an in depth, well written and detailed "
                f"introduction and conclusion to the research report based on the provided research data. "
                f"Do not include headers in the results.\n"
                f"You MUST include any relevant sources to the introduction and conclusion as markdown hyperlinks -"
                f"For example: 'This is a sample text. ([url website](url))'\n\n"
                f"{f'You must follow the guidelines provided: {guidelines}' if follow_guidelines else ''}\n"
                f"You MUST return nothing but a JSON in the following format (without json markdown):\n"
                f"{sample_json}\n\n",
            },
        ]

        response = await call_model(
            prompt,
            task.get("model"),
            response_format="json",
        )
        return response

    async def revise_headers(self, task: dict, headers: dict):
        prompt = [
            {
                "role": "system",
                "content": """You are a research writer. 
Your sole purpose is to revise the headers data based on the given guidelines.""",
            },
            {
                "role": "user",
                "content": f"""Your task is to revise the given headers JSON based on the guidelines given.
You are to follow the guidelines but the values should be in simple strings, ignoring all markdown syntax.
You must return nothing but a JSON in the same format as given in headers data.
Guidelines: {task.get("guidelines")}\n
Headers Data: {headers}\n
""",
            },
        ]

        response = await call_model(
            prompt,
            task.get("model"),
            response_format="json",
        )
        return {"headers": response}

    async def run(self, research_state: dict):
        """
        Generate the final research report by composing written sections, optionally streaming progress and verbose output, and (if requested) revising headers according to guidelines.
        
        This async method:
        - Calls write_sections(research_state) to produce the main report content.
        - If a websocket and stream_output are configured, sends status and verbose outputs via stream_output; otherwise prints messages via print_agent_output.
        - If task.follow_guidelines is true, requests header revisions via revise_headers and uses the returned headers.
        - Returns a merged dictionary containing the report content plus a top-level "headers" entry.
        
        Parameters:
            research_state (dict): Current research context. Must include a "task" dict with boolean keys "verbose" and "follow_guidelines". Other keys required by write_sections (e.g., query, data, task, guidelines) are passed through.
        
        Returns:
            dict: The final report content merged with a "headers" key (the headers are either generated by get_headers or replaced by revise_headers when follow_guidelines is true).
        """
        if self.websocket and self.stream_output:
            await self.stream_output(
                "logs",
                "writing_report",
                "Writing final research report based on research data...",
                self.websocket,
            )
        else:
            print_agent_output(
                "Writing final research report based on research data...",
                agent="WRITER",
            )

        research_layout_content = await self.write_sections(research_state)

        if research_state.get("task").get("verbose"):
            if self.websocket and self.stream_output:
                research_layout_content_str = json.dumps(
                    research_layout_content, indent=2
                )
                await self.stream_output(
                    "logs",
                    "research_layout_content",
                    research_layout_content_str,
                    self.websocket,
                )
            else:
                print_agent_output(research_layout_content, agent="WRITER")

        headers = self.get_headers(research_state)
        if research_state.get("task").get("follow_guidelines"):
            if self.websocket and self.stream_output:
                await self.stream_output(
                    "logs",
                    "rewriting_layout",
                    "Rewriting layout based on guidelines...",
                    self.websocket,
                )
            else:
                print_agent_output(
                    "Rewriting layout based on guidelines...", agent="WRITER"
                )
            headers = await self.revise_headers(
                task=research_state.get("task"), headers=headers
            )
            headers = headers.get("headers")

        return {**research_layout_content, "headers": headers}
