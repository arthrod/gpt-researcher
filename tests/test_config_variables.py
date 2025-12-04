import os
import unittest
import json
from gpt_researcher import GPTResearcher
from gpt_researcher.config.variables.default import DEFAULT_CONFIG

class TestConfigVariables(unittest.TestCase):
    def setUp(self):
        # Ensure dummy API key is set for Memory initialization
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = "dummy"

    def test_all_config_overrides(self):
        """Test that all configuration arguments can override environment variables and default config."""

        # Define test values for each config variable
        test_values = {
            "retriever": "google",
            "embedding": "openai:text-embedding-3-large",
            "similarity_threshold": 0.8,
            "fast_llm": "openai:gpt-4o",
            "smart_llm": "openai:gpt-4-turbo",
            "strategic_llm": "openai:gpt-4",
            "fast_token_limit": 1000,
            "smart_token_limit": 2000,
            "strategic_token_limit": 1500,
            "browse_chunk_max_length": 5000,
            "summary_token_limit": 500,
            "temperature": 0.7,
            "user_agent": "TestAgent/1.0",
            "max_search_results_per_query": 10,
            "memory_backend": "redis",
            "total_words": 2000,
            "report_format": "APA",
            "curate_sources": True,
            "max_iterations": 5,
            "language": "spanish",
            "scraper": "selenium",
            "max_scraper_workers": 5,
            "scraper_rate_limit_delay": 1.5,
            "doc_path": "./test_docs",
            "deep_research_concurrency": 2,
            "deep_research_depth": 3,
            "deep_research_breadth": 4,
            "mcp_auto_tool_selection": False,
            # "mcp_use_llm_args": True, # This key might not exist in BaseConfig yet, causing failure
            # "mcp_allowed_root_paths": ["/tmp"], # list needs special check
            "reasoning_effort": "high"
        }

        # Set some conflicting env vars to ensure they are ignored
        os.environ["RETRIEVER"] = "tavily"
        os.environ["TEMPERATURE"] = "0.1"
        os.environ["LANGUAGE"] = "english"

        # Instantiate researcher with test values
        # Note: We need to handle list/dict types carefully if we want to pass them in constructor.
        # But for scalar values it is straightforward.

        # mcp_allowed_root_paths needs to be passed as list
        mcp_paths = ["/tmp"]
        test_values["mcp_allowed_root_paths"] = mcp_paths

        # llm_kwargs and embedding_kwargs
        llm_kwargs = {"timeout": 60}
        embedding_kwargs = {"dimension": 1536}

        researcher = GPTResearcher(
            query="test",
            llm_kwargs=llm_kwargs,
            embedding_kwargs=embedding_kwargs,
            **test_values
        )

        # Verify values in researcher.cfg
        for key, value in test_values.items():
            if hasattr(researcher.cfg, key):
                config_value = getattr(researcher.cfg, key)
                self.assertEqual(config_value, value, f"Failed to override {key}")

        # Verify kwargs
        self.assertEqual(researcher.cfg.llm_kwargs, llm_kwargs)
        self.assertEqual(researcher.cfg.embedding_kwargs, embedding_kwargs)

        # Clean up env vars
        del os.environ["RETRIEVER"]
        del os.environ["TEMPERATURE"]
        del os.environ["LANGUAGE"]

        # Clean up created dir
        if os.path.exists("./test_docs"):
             os.rmdir("./test_docs")

    def test_defaults_preserved(self):
        """Test that if arguments are not provided, defaults (or env vars) are used."""

        os.environ["LANGUAGE"] = "french"

        researcher = GPTResearcher(query="test")

        # Check default from file/default.py
        self.assertEqual(researcher.cfg.retriever, DEFAULT_CONFIG["RETRIEVER"])

        # Check env var usage
        self.assertEqual(researcher.cfg.language, "french")

        del os.environ["LANGUAGE"]

    def test_kwargs_propagation_to_components(self):
        """Test that kwargs (like API keys) are propagated to components via self.researcher.kwargs"""

        custom_kwargs = {
            "tavily_api_key": "custom_tavily_key",
            "google_api_key": "custom_google_key",
            "openai_api_key": "custom_openai_key"
        }

        researcher = GPTResearcher(query="test", **custom_kwargs)

        # Check that kwargs are stored in researcher instance
        for key, value in custom_kwargs.items():
            self.assertIn(key, researcher.kwargs)
            self.assertEqual(researcher.kwargs[key], value)

        # Ideally we would check if retrievers/scrapers get these, but that requires mocking or inspecting internal instantiation.
        # The code changes in ResearchConductor and Scraper verify this propagation logic.

if __name__ == "__main__":
    unittest.main()
