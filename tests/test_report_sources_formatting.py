"""Tests for report sources formatting fix."""


# Add project root to path
import sys

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.append(str(Path(__file__).parent.parent))

from gpt_researcher.skills.writer import ReportGenerator


class MockConfig:
    """Mock configuration object."""

    def __init__(self, append_sources=True):
        self.append_sources = append_sources
        self.agent_role = None  # Add required attribute


class MockResearcher:
    """Mock researcher object for testing."""

    def __init__(self, append_sources=True, sources=None):
        self.cfg = MockConfig(append_sources)
        self.research_sources = sources or []
        self.verbose = False
        self.websocket = None
        self.query = "test query"
        self.kwargs = {}
        # Add required attributes for ReportGenerator
        self.role = "default researcher"
        self.report_type = "research_report"
        self.report_source = "web"
        self.tone = None
        self.headers = {}
        self.context = []
        self.research_images = []

    def get_research_images(self):
        """Mock method to get research images."""
        return self.research_images

    def get_research_sources(self):
        """Mock method to get research sources."""
        return self.research_sources

    def add_costs(self, cost):
        """Mock method to track costs."""
        pass


@pytest.mark.asyncio
async def test_sources_formatting_with_proper_newlines():
    """Test that sources are formatted with proper newlines."""

    # Create mock researcher with sources
    sources = [
        {"id": 1, "title": "Article 1", "url": "https://example.com/1"},
        {"id": 2, "title": "Article 2", "url": "https://example.com/2"},
        {"id": 3, "url": "https://example.com/3"},  # No title
    ]
    researcher = MockResearcher(append_sources=True, sources=sources)

    # Create report generator
    generator = ReportGenerator(researcher)

    # Mock the generate_report function
    mock_report_content = "This is the main report content."

    with patch(
        "gpt_researcher.skills.writer.generate_report", new_callable=AsyncMock
    ) as mock_gen:
        mock_gen.return_value = mock_report_content

        # Generate report
        report = await generator.write_report()

    # Verify formatting
    assert report.startswith("This is the main report content.")
    assert "\n\n\nSources:\n\n" in report
    assert "[1] Article 1 - https://example.com/1" in report
    assert "[2] Article 2 - https://example.com/2" in report
    assert "[3] https://example.com/3 - https://example.com/3" in report
    assert report.endswith("\n")


@pytest.mark.asyncio
async def test_sources_with_missing_keys():
    """Test handling of sources with missing keys."""

    # Create sources with various missing keys
    sources = [
        {"id": 1, "title": "Valid", "url": "https://example.com/1"},
        {"title": "Missing ID", "url": "https://example.com/2"},  # Missing id
        {"id": 3, "title": "Missing URL"},  # Missing url
        {"id": 4},  # Missing both title and url
        {},  # Empty source
    ]
    researcher = MockResearcher(append_sources=True, sources=sources)

    generator = ReportGenerator(researcher)

    with patch(
        "gpt_researcher.skills.writer.generate_report", new_callable=AsyncMock
    ) as mock_gen:
        mock_gen.return_value = "Main report."

        report = await generator.write_report()

    # Only the valid source should be included
    assert "[1] Valid - https://example.com/1" in report
    assert "[2]" not in report  # Missing id
    assert "[3]" not in report  # Missing url
    assert "[4]" not in report  # Missing title and url


@pytest.mark.asyncio
async def test_no_sources_appended_when_disabled():
    """Test that sources are not appended when append_sources is False."""

    sources = [{"id": 1, "title": "Article", "url": "https://example.com/1"}]
    researcher = MockResearcher(append_sources=False, sources=sources)

    generator = ReportGenerator(researcher)

    with patch(
        "gpt_researcher.skills.writer.generate_report", new_callable=AsyncMock
    ) as mock_gen:
        mock_gen.return_value = "Main report content."

        report = await generator.write_report()

    # Sources should not be appended
    assert report == "Main report content."
    assert "Sources:" not in report


@pytest.mark.asyncio
async def test_safe_getattr_for_append_sources():
    """Test safe access to append_sources attribute."""

    class MinimalConfig:
        """Config without append_sources attribute."""

        agent_role = None  # Add required attribute

    researcher = MockResearcher(append_sources=True)
    researcher.cfg = MinimalConfig()  # Replace with config missing append_sources
    researcher.research_sources = [
        {"id": 1, "title": "Test", "url": "https://example.com"}
    ]

    generator = ReportGenerator(researcher)

    with patch(
        "gpt_researcher.skills.writer.generate_report", new_callable=AsyncMock
    ) as mock_gen:
        mock_gen.return_value = "Report content."

        # Should not raise AttributeError
        report = await generator.write_report()

    # Sources should not be appended (defaults to False)
    assert report == "Report content."
    assert "Sources:" not in report


@pytest.mark.asyncio
async def test_report_ending_with_whitespace():
    """Test handling of reports ending with various whitespace."""

    sources = [{"id": 1, "title": "Source", "url": "https://example.com"}]
    researcher = MockResearcher(append_sources=True, sources=sources)

    generator = ReportGenerator(researcher)

    # Test various report endings
    test_cases = [
        "Report with spaces   ",
        "Report with newlines\n\n\n",
        "Report with tabs\t\t",
        "Report with mixed whitespace  \n\t  ",
    ]

    for test_content in test_cases:
        with patch(
            "gpt_researcher.skills.writer.generate_report", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = test_content

            report = await generator.write_report()

        # Should have consistent formatting regardless of trailing whitespace
        assert not report.startswith("\n")
        assert "\n\n\nSources:\n\n" in report
        assert report.endswith("\n")
        assert not report.endswith("\n\n")


@pytest.mark.asyncio
async def test_empty_sources_list():
    """Test behavior with empty sources list."""

    researcher = MockResearcher(append_sources=True, sources=[])

    generator = ReportGenerator(researcher)

    with patch(
        "gpt_researcher.skills.writer.generate_report", new_callable=AsyncMock
    ) as mock_gen:
        mock_gen.return_value = "Report content."

        report = await generator.write_report()

    # Should not append sources section for empty list
    assert report == "Report content."
    assert "Sources:" not in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
