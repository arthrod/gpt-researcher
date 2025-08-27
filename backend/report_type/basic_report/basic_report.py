from typing import Any

from fastapi import WebSocket

from gpt_researcher import GPTResearcher


class BasicReport:
    def __init__(
        self,
        query: str,
        query_domains: list,
        report_type: str,
        report_source: str,
        source_urls,
        document_urls,
        tone: Any,
        config_path: str,
        websocket: WebSocket,
        headers=None,
        mcp_configs=None,
        mcp_strategy=None,
    ):
        """
        Initialize the BasicReport wrapper and construct an internal GPTResearcher configured to run the requested research and generate a report.
        
        Parameters:
            query (str): Search or research prompt driving the report.
            query_domains (list): Domains or areas to constrain the research (e.g., ["biology", "finance"]).
            report_type (str): The format or style of report to produce (e.g., "summary", "detailed").
            report_source (str): Identifier for the source or origin of the report request.
            source_urls: Iterable of source URLs to include or prioritize during research.
            document_urls: Iterable of document URLs to include as source material.
            tone (Any): Tone/style guidance to apply to the generated report (format depends on upstream code).
            config_path (str): Filesystem path to configuration used by the researcher.
            headers (optional): HTTP headers to attach to outbound requests; defaults to an empty dict when None.
            mcp_configs (optional): Optional MCP configuration object or mapping passed through to GPTResearcher when provided.
            mcp_strategy (optional): Optional MCP strategy identifier or object passed through to GPTResearcher when provided.
        
        Side effects:
            - Creates and assigns a GPTResearcher instance to self.gpt_researcher using the provided parameters.
        """
        self.query = query
        self.query_domains = query_domains
        self.report_type = report_type
        self.report_source = report_source
        self.source_urls = source_urls
        self.document_urls = document_urls
        self.tone = tone
        self.config_path = config_path
        self.websocket = websocket
        self.headers = headers or {}

        # Initialize researcher with optional MCP parameters
        gpt_researcher_params = {
            "query": self.query,
            "query_domains": self.query_domains,
            "report_type": self.report_type,
            "report_source": self.report_source,
            "source_urls": self.source_urls,
            "document_urls": self.document_urls,
            "tone": self.tone,
            "config_path": self.config_path,
            "websocket": self.websocket,
            "headers": self.headers,
        }

        # Add MCP parameters if provided
        if mcp_configs is not None:
            gpt_researcher_params["mcp_configs"] = mcp_configs
        if mcp_strategy is not None:
            gpt_researcher_params["mcp_strategy"] = mcp_strategy

        self.gpt_researcher = GPTResearcher(**gpt_researcher_params)

    async def run(self):
        """
        Asynchronously runs the research and report generation process using the encapsulated GPTResearcher.
        
        Performs research (awaits conduct_research) and then generates a report (awaits write_report), returning the generated report object produced by GPTResearcher.
        """
        await self.gpt_researcher.conduct_research()
        report = await self.gpt_researcher.write_report()
        return report
