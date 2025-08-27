<<<<<<< HEAD
from typing import TypedDict, List
=======
from typing import TypedDict
>>>>>>> newdev


class ResearchState(TypedDict):
    task: dict
    initial_research: str
    sections: list[str]
    research_data: list[dict]
    human_feedback: str
    # Report layout
    title: str
    headers: dict
    date: str
    table_of_contents: str
    introduction: str
    conclusion: str
    sources: list[str]
    report: str
