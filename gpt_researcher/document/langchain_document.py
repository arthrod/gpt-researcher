
from langchain_core.documents import Document


# Supports the base Document class from langchain
# - https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/documents/base.py
class LangChainDocumentLoader:
    def __init__(self, documents: list[Document]):
        self.documents = documents

    async def load(self, metadata_source_index="title") -> list[dict[str, str]]:
        docs = []
        for document in self.documents:
            docs.append(
                {
                    "raw_content": document.page_content,
                    "url": document.metadata.get(metadata_source_index, ""),
                }
            )
        return docs
