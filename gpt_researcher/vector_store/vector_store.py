"""
Wrapper for langchain vector store
"""
from typing import List, Dict

from langchain.docstore.document import Document
from langchain.vectorstores import VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorStoreWrapper:
    """
    A Wrapper for LangchainVectorStore to handle GPT-Researcher Document Type
    """
    def __init__(self, vector_store: VectorStore):
        """
        Initialize the wrapper with an underlying Langchain VectorStore.
        
        Stores the provided VectorStore instance on self.vector_store for all subsequent ingestion and similarity-search operations.
        """
        self.vector_store = vector_store

    def load(self, documents):
        """
        Ingest a list of GPT-Researcher-style records into the wrapped vector store.
        
        Converts input records to LangChain Document objects, splits them into text chunks using the wrapper's splitter, and adds the resulting documents to the underlying vector store.
        
        Parameters:
            documents (List[Dict[str, str]]): Iterable of records where each item must contain:
                - "raw_content": the full text to index
                - "url": the source identifier stored as the Document's "source" metadata
        
        Returns:
            None
        """
        langchain_documents = self._create_langchain_documents(documents)
        splitted_documents = self._split_documents(langchain_documents)
        self.vector_store.add_documents(splitted_documents)

    def _create_langchain_documents(self, data: List[Dict[str, str]]) -> List[Document]:
        """
        Convert a list of GPT-Researcher-style dicts into Langchain Document objects.
        
        Each input dict must contain the keys:
        - "raw_content": string used as the Document's page_content.
        - "url": string stored in the Document's metadata under the "source" key.
        
        Parameters:
            data (List[Dict[str, str]]): Iterable of document dicts with "raw_content" and "url".
        
        Returns:
            List[Document]: Langchain Document instances with page_content and metadata populated.
        """
        return [Document(page_content=item["raw_content"], metadata={"source": item["url"]}) for item in data]

    def _split_documents(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """
        Split documents into smaller chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return text_splitter.split_documents(documents)

    async def asimilarity_search(self, query, k, filter):
        """Return query by vector store"""
        results = await self.vector_store.asimilarity_search(query=query, k=k, filter=filter)
        return results
