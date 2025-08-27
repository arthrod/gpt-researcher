<<<<<<< HEAD
from typing import TypedDict
=======
from typing import TypedDict, List
>>>>>>> 1027e1d0 (Fix linting issues)


class ResearchState(TypedDict):
    task: dict
    initial_research: str
    sections: list[str]
    research_data: list[dict]
    # Report layout
    title: str
    headers: dict
    date: str
    table_of_contents: str
    introduction: str
    conclusion: str
    sources: list[str]
    report: str
