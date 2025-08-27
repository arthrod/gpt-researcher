<<<<<<< HEAD
from typing import Dict, List
=======
from typing import ClassVar
>>>>>>> newdev

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever


class SearchAPIRetriever(BaseRetriever):
    """Search API retriever."""

    pages: ClassVar[list[dict]] = []

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        docs = [
            Document(
                page_content=page.get("raw_content", ""),
                metadata={
                    "title": page.get("title", ""),
                    "source": page.get("url", ""),
                },
            )
            for page in self.pages
        ]

        return docs


class SectionRetriever(BaseRetriever):
    """
    SectionRetriever:
    This class is used to retrieve sections while avoiding redundant subtopics.
    """

    sections: ClassVar[list[dict]] = []
    """
    sections example:
    [
        {
            "section_title": "Example Title",
            "written_content": "Example content"
        },
        ...
    ]
    """

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
<<<<<<< HEAD
    ) -> List[Document]:

        """
        Return all stored sections as LangChain Documents.
        
        Builds a Document for each item in self.sections using the section's
        `written_content` as the document text and `section_title` as metadata.
        
        Parameters:
            query (str): Accepted for Retriever API compatibility but ignored by this implementation.
        
        Returns:
            List[Document]: Documents created from self.sections where each document's
            page_content is the section's `written_content` and metadata contains
            "section_title".
        """
=======
    ) -> list[Document]:
>>>>>>> newdev
        docs = [
            Document(
                page_content=page.get("written_content", ""),
                metadata={
                    "section_title": page.get("section_title", ""),
                },
            )
            for page in self.sections  # Changed 'self.pages' to 'self.sections'
        ]

        return docs
