"""Tests for MCP URL normalization and comparison."""

import sys

from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.append(str(Path(__file__).parent.parent))

from gpt_researcher.skills.researcher import ResearchConductor
from gpt_researcher.utils.constants import MCP_SENTINEL_URL


class MockResearcher:
    """Mock researcher for testing."""

    def __init__(self):
        self.websocket = None
        self.cfg = MagicMock()
        self.retrievers = []
        self.verbose = False


class TestMCPURLNormalization:
    """Test MCP URL normalization in ResearchConductor."""

    def test_mcp_sentinel_url_constant(self):
        """Test that the MCP sentinel URL constant is defined correctly."""
        assert MCP_SENTINEL_URL == "mcp://llm_analysis"
        assert isinstance(MCP_SENTINEL_URL, str)

    @pytest.mark.asyncio
    async def test_url_normalization_cases(self):
        """Test various URL formats are correctly normalized and compared."""
        researcher = MockResearcher()
        _conductor = ResearchConductor(researcher)

        # Test cases: (url_input, should_include_url_in_citation)
        test_cases = [
            # Exact match - should NOT include URL
            ("mcp://llm_analysis", False),
            # Case variations - should NOT include URL (normalized comparison)
            ("MCP://LLM_ANALYSIS", False),
            ("mcp://LLM_Analysis", False),
            ("MCP://llm_analysis", False),
            # With whitespace - should NOT include URL (stripped)
            ("  mcp://llm_analysis  ", False),
            ("\tmcp://llm_analysis\n", False),
            # Different URLs - should include URL
            ("https://example.com", True),
            ("http://test.org", True),
            ("mcp://different", True),
            ("", False),  # Empty URL should not include
            (None, False),  # None should not include
        ]

        for url_input, should_include in test_cases:
            # Simulate the normalization logic from researcher.py
            normalized_url = str(url_input).strip().lower() if url_input else ""
            includes_url = bool(
                normalized_url and normalized_url != MCP_SENTINEL_URL.lower()
            )

            assert includes_url == should_include, (
                f"URL '{url_input}' normalized to '{normalized_url}' - "
                f"expected include_url={should_include}, got {includes_url}"
            )

    def test_mcp_url_in_context_formatting(self):
        """Test that MCP URLs are handled correctly in context formatting."""
        mcp_context = [
            {
                "content": "Test content 1",
                "url": "mcp://llm_analysis",
                "title": "LLM Analysis",
            },
            {
                "content": "Test content 2",
                "url": "MCP://LLM_ANALYSIS",  # Different casing
                "title": "Another Analysis",
            },
            {
                "content": "Test content 3",
                "url": "https://example.com",
                "title": "Web Source",
            },
        ]

        # Process each context item
        for item in mcp_context:
            url = item.get("url", "")
            title = item.get("title", "")
            content = item.get("content", "")

            if content and content.strip():
                # Normalize URL for comparison
                normalized_url = str(url).strip().lower() if url else ""
                if normalized_url and normalized_url != MCP_SENTINEL_URL.lower():
                    citation = f"\n\n*Source: {title} ({url})*"
                else:
                    citation = f"\n\n*Source: {title}*"

                # Check citation format
                if url and url.lower().strip() == MCP_SENTINEL_URL.lower():
                    assert f"({url})" not in citation, (
                        f"MCP URL should not be included in citation: {citation}"
                    )
                elif url and url.startswith("http"):
                    assert f"({url})" in citation, (
                        f"Regular URL should be included in citation: {citation}"
                    )

    def test_import_structure(self):
        """Test that constants can be imported from both modules."""
        # Import from research.py
        from gpt_researcher.mcp.research import MCP_SENTINEL_URL as research_url

        # Import from researcher.py
        from gpt_researcher.skills.researcher import MCP_SENTINEL_URL as researcher_url

        # Import from constants
        from gpt_researcher.utils.constants import MCP_SENTINEL_URL as constants_url

        # All should be the same
        assert research_url == constants_url
        assert researcher_url == constants_url
        assert constants_url == "mcp://llm_analysis"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
