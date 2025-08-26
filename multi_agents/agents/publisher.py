from .utils.file_formats import write_md_to_pdf, write_md_to_word, write_text_to_md
from .utils.views import print_agent_output


class PublisherAgent:
<<<<<<< HEAD
    def __init__(
        self, output_dir: str, websocket=None, stream_output=None, headers=None
    ):
=======
    def __init__(self, output_dir: str, websocket=None, stream_output=None, headers=None):
        """
        Initialize the PublisherAgent.
        
        Parameters:
            output_dir (str): Destination directory for published files; leading/trailing whitespace is stripped.
            headers (dict, optional): Default header values (e.g., title, date, introduction) used when generating the report. Defaults to an empty dict.
        
        Notes:
            Stores websocket and stream_output on the instance for optional live streaming; these are passed through without modification.
        """
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        self.websocket = websocket
        self.stream_output = stream_output
        self.output_dir = output_dir.strip()
        self.headers = headers or {}

<<<<<<< HEAD
    async def publish_research_report(
        self, research_state: dict, publish_formats: dict
    ):
=======
    async def publish_research_report(self, research_state: dict, publish_formats: dict):
<<<<<<< HEAD
>>>>>>> 1027e1d0 (Fix linting issues)
=======
        """
        Generate the final report layout from research state and write it to the requested formats.
        
        Parameters:
            research_state (dict): Research data and metadata used to build the report layout. Must follow the shape expected by generate_layout (e.g., keys like "research_data", "sources", "headers").
            publish_formats (dict): Mapping of output formats to truthy values indicating which formats to produce (supported keys include "pdf", "docx", "markdown").
        
        Returns:
            str: The generated Markdown layout string.
        """
>>>>>>> 9a0c4dfe (📝 Add docstrings to `enhancements/highlevel-instructions`)
        layout = self.generate_layout(research_state)
        await self.write_report_by_formats(layout, publish_formats)

        return layout

    def generate_layout(self, research_state: dict):
        """
        Generate a Markdown report layout from a research state dictionary.
        
        Takes the provided research_state and assembles a Markdown-formatted report string. Expects research_state to contain:
        - "research_data": list of sections; each item may be a string or a dict (dict values are appended in iteration order).
        - "sources": iterable of reference strings included under a References section.
        - "headers": mapping of section titles (e.g., "title", "introduction", "table_of_contents", "conclusion", "references", "date").
        - Optional top-level keys used as content: "date", "introduction", "table_of_contents", "conclusion".
        
        Returns:
            str: The generated Markdown layout containing title, date line, introduction, table of contents, assembled sections, conclusion, and references.
        """
        sections = []
        for subheader in research_state.get("research_data", []):
            if isinstance(subheader, dict):
                # Handle dictionary case
                for _key, value in subheader.items():
                    sections.append(f"{value}")
            else:
                # Handle string case
                sections.append(f"{subheader}")

<<<<<<< HEAD
        sections_text = "\n\n".join(sections)
        references = "\n".join(
            f"{reference}" for reference in research_state.get("sources", [])
        )
=======
        sections_text = '\n\n'.join(sections)
        references = '\n'.join(f"{reference}" for reference in research_state.get("sources", []))
>>>>>>> 1027e1d0 (Fix linting issues)
        headers = research_state.get("headers", {})
        layout = f"""# {headers.get("title")}
#### {headers.get("date")}: {research_state.get("date")}

## {headers.get("introduction")}
{research_state.get("introduction")}

## {headers.get("table_of_contents")}
{research_state.get("table_of_contents")}

{sections_text}

## {headers.get("conclusion")}
{research_state.get("conclusion")}

## {headers.get("references")}
{references}
"""
        return layout

    async def write_report_by_formats(self, layout: str, publish_formats: dict):
        if publish_formats.get("pdf"):
            await write_md_to_pdf(layout, self.output_dir)
        if publish_formats.get("docx"):
            await write_md_to_word(layout, self.output_dir)
        if publish_formats.get("markdown"):
            await write_text_to_md(layout, self.output_dir)

    async def run(self, research_state: dict):
        """
        Run the publisher: generate and publish the final research report, and return the generated report.
        
        Parameters:
            research_state (dict): State produced by the research process. Expected to contain a "task" mapping with a "publish_formats" key and the data required by publish_research_report.
        
        Returns:
            dict: {"report": layout_str} where layout_str is the generated report layout returned by publish_research_report.
        
        Notes:
            - Emits a status message either via the configured stream_output/websocket or via print_agent_output.
            - publish_research_report may write output files (PDF, DOCX, Markdown) according to publish_formats.
        """
        task = research_state.get("task")
        publish_formats = task.get("publish_formats")
        if self.websocket and self.stream_output:
<<<<<<< HEAD
            await self.stream_output(
                "logs",
                "publishing",
                "Publishing final research report based on retrieved data...",
                self.websocket,
            )
=======
            await self.stream_output("logs", "publishing", "Publishing final research report based on retrieved data...", self.websocket)
>>>>>>> 1027e1d0 (Fix linting issues)
        else:
            print_agent_output(
                output="Publishing final research report based on retrieved data...",
                agent="PUBLISHER",
            )
        final_research_report = await self.publish_research_report(
            research_state, publish_formats
        )
        return {"report": final_research_report}
