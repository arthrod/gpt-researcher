import uuid

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool, tool
from langchain_community.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from gpt_researcher.config.config import Config
from gpt_researcher.memory import Memory
from gpt_researcher.utils.llm import get_llm


class ChatAgentWithMemory:
    def __init__(self, report: str, config_path, headers, vector_store=None):
        self.report = report
        self.headers = headers
        self.config = Config(config_path)
        self.vector_store = vector_store
        self.graph = self.create_agent()

    def create_agent(self):
        """
        Create and configure a React Agent Graph backed by an LLM and an in-memory vector store.
        
        Builds an LLM provider from configuration and instantiates a React Agent Graph with a retrieval tool and memory checkpointer. If this instance has no preexisting vector store, the method will:
        - split and embed self.report,
        - create and store self.chat_config (with a unique thread_id),
        - set self.embedding and self.vector_store, and
        - index the document chunks into the in-memory vector store.
        
        Notes:
        - Temperature and max_tokens are only added to the LLM init kwargs when the selected model supports temperature.
        - Returns the created agent graph instance.
        - Side effects: may set self.chat_config, self.embedding, and self.vector_store.
        """
        cfg = Config()

        # Retrieve LLM using get_llm with settings from config
        # Avoid passing temperature for models that do not support it
        from gpt_researcher.llm_provider.generic.base import (
            NO_SUPPORT_TEMPERATURE_MODELS,
        )

        llm_init_kwargs = {
            "llm_provider": cfg.smart_llm_provider,
            "model": cfg.smart_llm_model,
            **self.config.llm_kwargs,
        }

        if cfg.smart_llm_model not in NO_SUPPORT_TEMPERATURE_MODELS:
            llm_init_kwargs["temperature"] = 0.35
            llm_init_kwargs["max_tokens"] = cfg.smart_token_limit

        provider = get_llm(**llm_init_kwargs).llm

        # If vector_store is not initialized, process documents and add to vector_store
        if not self.vector_store:
            documents = self._process_document(self.report)
            self.chat_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            self.embedding = Memory(
                cfg.embedding_provider, cfg.embedding_model, **cfg.embedding_kwargs
            ).get_embeddings()
            self.vector_store = InMemoryVectorStore(self.embedding)
            self.vector_store.add_texts(documents)

        # Create the React Agent Graph with the configured provider
        graph = create_react_agent(
            provider,
            tools=[self.vector_store_tool(self.vector_store)],
            checkpointer=MemorySaver(),
        )

        return graph

    def vector_store_tool(self, vector_store) -> Tool:
<<<<<<< HEAD
        """
        Create a retrieval tool that queries the provided vector store for contextual documents.
        
        The returned tool is callable by the agent and accepts a single string `query`. It builds a retriever
        from the given vector store (top-k = 4) and returns the retriever's results for the query.
        """
=======
        """Create Vector Store Tool"""

>>>>>>> newdev
        @tool
        def retrieve_info(query):
            """
            Retrieve relevant contextual documents from the vector store for a user query.
            
            Performs a k=4 retrieval against the enclosing vector_store and returns the retriever's result (contextual documents or snippets) for use by the agent.
            
            Parameters:
                query (str): The user query or prompt to search for relevant context.
            
            Returns:
                The retriever's response containing the retrieved contexts (format depends on the vector store implementation).
            """
            retriever = vector_store.as_retriever(k=4)
            return retriever.invoke(query)

        return retrieve_info

    def _process_document(self, report):
        """
        Split a long report string into smaller text chunks suitable for embedding and indexing.
        
        This uses a RecursiveCharacterTextSplitter configured to produce chunks of up to 1024 characters with a 20-character overlap to preserve context between chunks.
        
        Parameters:
            report (str): The full report text to split.
        
        Returns:
            List[str]: A list of text chunks extracted from the report.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        documents = text_splitter.split_text(report)
        return documents

    async def chat(self, message, websocket):
        """Chat with React Agent"""
        message = f"""
         You are GPT Researcher, a autonomous research agent created by an open source community at https://github.com/assafelovic/gpt-researcher, homepage: https://gptr.dev.
         To learn more about GPT Researcher you can suggest to check out: https://docs.gptr.dev.

         This is a chat message between the user and you: GPT Researcher.
         The chat is about a research reports that you created. Answer based on the given context and report.
         You must include citations to your answer based on the report.

         Report: {self.report}
         User Message: {message}
        """
        inputs = {"messages": [("user", message)]}
        response = await self.graph.ainvoke(inputs, config=self.chat_config)
        ai_message = response["messages"][-1].content
        if websocket is not None:
            await websocket.send_json({"type": "chat", "content": ai_message})

    def get_context(self):
        """return the current context of the chat"""
        return self.report
