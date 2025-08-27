import asyncio

import nest_asyncio

from gpt_researcher import GPTResearcher

nest_asyncio.apply()


<<<<<<< HEAD

async def get_report(query: str, report_type: str, custom_prompt: str = None):
    """
    Run an end-to-end research workflow and produce a report plus related metadata.
    
    Performs asynchronous research using GPTResearcher constructed from `query` and `report_type`, generates a written report (optionally using `custom_prompt`), and returns supplementary data collected during the workflow.
    
    Parameters:
        query (str): The research question or topic to investigate.
        report_type (str): Specifies the report style or format to produce (e.g., "research_report"); affects how GPTResearcher conducts research and writes the report.
        custom_prompt (str, optional): If provided, overrides or customizes the prompt used when generating the final report.
    
    Returns:
        tuple: (report, research_context, research_costs, research_images, research_sources)
            - report (str): Generated report text.
            - research_context (Any): Internal context/state gathered during research (implementation-specific).
            - research_costs (Any): Cost/usage information produced by the researcher.
            - research_images (list): Collected images or image metadata related to the research.
            - research_sources (list): Source citations or references discovered during research.
    
    Notes:
        - This function performs asynchronous I/O and may incur external API/LLM calls and associated costs.
    """
    researcher = GPTResearcher(query, report_type)
    research_result = await researcher.conduct_research()
=======
async def get_report(query: str, report_type: str, custom_prompt: str | None = None):
    researcher = GPTResearcher(query, report_type)
    _research_result = await researcher.conduct_research()
>>>>>>> newdev

    # Generate report with optional custom prompt
    report = await researcher.write_report(custom_prompt=custom_prompt)

    # Get additional information
    research_context = researcher.get_research_context()
    research_costs = researcher.get_costs()
    research_images = researcher.get_research_images()
    research_sources = researcher.get_research_sources()

    return report, research_context, research_costs, research_images, research_sources


if __name__ == "__main__":
    query = "Should I invest in Nvidia?"
    report_type = "research_report"

    # Standard report
    report, context, costs, images, sources = asyncio.run(
        get_report(query, report_type)
    )

    print("Standard Report:")
    print(report)

    # Custom report with specific formatting requirements
    custom_prompt = "Answer in short, 2 paragraphs max without citations. Focus on the most important facts for investors."
<<<<<<< HEAD
    custom_report, _, _, _, _ = asyncio.run(get_report(query, report_type, custom_prompt))
=======
    custom_report, _, _, _, _ = asyncio.run(
        get_report(query, report_type, custom_prompt)
    )
>>>>>>> newdev

    print("\nCustomized Short Report:")
    print(custom_report)

    print("\nResearch Costs:")
    print(costs)
    print("\nNumber of Research Images:")
    print(len(images))
    print("\nNumber of Research Sources:")
    print(len(sources))
