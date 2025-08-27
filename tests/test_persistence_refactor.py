"""Tests for the refactored persistence layer."""

import asyncio

# Add project root to path
import sys

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.append(str(Path(__file__).parent.parent))

from gpt_researcher.persistence import ScraperRepository
from gpt_researcher.scraper.utils_refactored import (
    ScraperDataManager,
)


class MockEmbeddings:
    """Mock embeddings class for testing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate fake embeddings."""
        return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.fixture
def mock_embeddings():
    """Fixture for mock embeddings."""
    return MockEmbeddings()


@pytest.fixture
def sample_records():
    """Fixture for sample scraped records."""
    return [
        {
            "url": f"https://example.com/page{i}",
            "title": f"Page {i}",
            "raw_content": f"Content of page {i} with some text...",
        }
        for i in range(1, 6)
    ]


@pytest.mark.asyncio
async def test_repository_initialization():
    """Test repository initialization and database setup."""
    db_url = "postgresql+asyncpg://test:test@localhost/test"

    with patch(
        "gpt_researcher.persistence.repository.create_async_engine"
    ) as mock_engine:
        mock_engine.return_value = AsyncMock()

        repository = ScraperRepository(
            db_url=db_url, embeddings=MockEmbeddings(), pool_size=10
        )

        assert repository.embeddings is not None
        assert repository.engine is not None
        mock_engine.assert_called_once()


@pytest.mark.asyncio
async def test_async_embedding_generation():
    """Test that embeddings are generated asynchronously."""
    repository = ScraperRepository(
        db_url="postgresql+asyncpg://test:test@localhost/test",
        embeddings=MockEmbeddings(),
    )

    # Test embedding generation
    content = "Test content for embedding"
    embedding = await repository._get_embedding_async(content)

    assert embedding is not None
    assert len(embedding) == 3
    assert embedding == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_batch_processing(sample_records):
    """Test batch processing of records."""
    with patch(
        "gpt_researcher.persistence.repository.create_async_engine"
    ) as mock_engine:
        # Create a proper engine mock
        mock_engine_instance = AsyncMock()
        mock_engine.return_value = mock_engine_instance

        # Mock the async sessionmaker to return a callable that creates sessions
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.execute = AsyncMock()

        def mock_session_factory():
            return mock_session

        with patch(
            "gpt_researcher.persistence.repository.async_sessionmaker"
        ) as mock_sessionmaker:
            mock_sessionmaker.return_value = mock_session_factory

            repository = ScraperRepository(
                db_url="postgresql+asyncpg://test:test@localhost/test",
                embeddings=MockEmbeddings(),
            )

            # Mock the get_session method directly to avoid complex async context manager mocking
            async def mock_get_session():
                yield mock_session

            repository.get_session = AsyncMock(side_effect=mock_get_session)
            repository.get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            repository.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

            # Test saving records in batches
            _saved_count = await repository.save_records(
                records=sample_records,
                batch_size=2,  # Small batch size for testing
            )

            # Verify the session was used
            assert repository.get_session.called


@pytest.mark.asyncio
async def test_scraper_data_manager_context():
    """Test ScraperDataManager as async context manager."""
    with patch(
        "gpt_researcher.persistence.repository.create_async_engine"
    ) as mock_engine:
        # Create proper async engine mock with begin context manager
        mock_engine_instance = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.run_sync = AsyncMock()

        # Mock the begin context manager
        mock_engine_instance.begin = AsyncMock()
        mock_engine_instance.begin.return_value.__aenter__ = AsyncMock(
            return_value=mock_conn
        )
        mock_engine_instance.begin.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_engine_instance.dispose = AsyncMock()

        mock_engine.return_value = mock_engine_instance

        # Mock async_sessionmaker
        mock_session = AsyncMock()

        def mock_session_factory():
            return mock_session

        with patch(
            "gpt_researcher.persistence.repository.async_sessionmaker"
        ) as mock_sessionmaker:
            mock_sessionmaker.return_value = mock_session_factory

            db_url = "postgresql+asyncpg://test:test@localhost/test"

            async with ScraperDataManager(db_url) as manager:
                assert manager._initialized is True
                assert manager.repository is not None

            # After context exit, repository should be closed
            mock_engine_instance.dispose.assert_called()


@pytest.mark.asyncio
async def test_filter_existing_urls():
    """Test filtering of existing URLs."""
    with patch(
        "gpt_researcher.persistence.repository.create_async_engine"
    ) as mock_engine:
        mock_engine.return_value = AsyncMock()

        manager = ScraperDataManager("postgresql+asyncpg://test:test@localhost/test")

        # Mock the repository method
        existing_docs = [
            MagicMock(url="https://example.com/page1"),
            MagicMock(url="https://example.com/page3"),
        ]
        manager.repository.get_documents_batch = AsyncMock(return_value=existing_docs)

        # Test filtering
        all_urls = [
            "https://example.com/page1",
            "https://example.com/page2",
            "https://example.com/page3",
            "https://example.com/page4",
        ]

        new_urls = await manager.filter_new_urls(all_urls)

        assert len(new_urls) == 2
        assert "https://example.com/page2" in new_urls
        assert "https://example.com/page4" in new_urls


@pytest.mark.asyncio
async def test_performance_improvement():
    """Test that the refactored version doesn't block the event loop."""
    import time

    class SlowEmbeddings:
        """Embeddings that simulate slow blocking operations."""

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            # Simulate slow blocking operation
            time.sleep(0.1)  # 100ms per call
            return [[0.1] * 10 for _ in texts]

    with patch(
        "gpt_researcher.persistence.repository.create_async_engine"
    ) as mock_engine:
        mock_engine.return_value = AsyncMock()

        repository = ScraperRepository(
            db_url="postgresql+asyncpg://test:test@localhost/test",
            embeddings=SlowEmbeddings(),
        )

        # This should not block the event loop
        start_time = asyncio.get_event_loop().time()

        # Generate embeddings for multiple documents concurrently
        tasks = [repository._get_embedding_async(f"Content {i}") for i in range(5)]

        results = await asyncio.gather(*tasks)

        elapsed_time = asyncio.get_event_loop().time() - start_time

        # All embeddings should be generated
        assert len(results) == 5
        assert all(r is not None for r in results)

        # Should complete faster than sequential execution (5 * 0.1 = 0.5s)
        # Due to thread pool execution, should be closer to 0.1-0.2s
        assert elapsed_time < 0.5  # Much faster than sequential


@pytest.mark.asyncio
async def test_error_handling():
    """Test proper error handling in repository."""

    class FailingEmbeddings:
        """Embeddings that raise errors."""

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise Exception("Embedding generation failed")

    with patch(
        "gpt_researcher.persistence.repository.create_async_engine"
    ) as mock_engine:
        mock_engine.return_value = AsyncMock()

        repository = ScraperRepository(
            db_url="postgresql+asyncpg://test:test@localhost/test",
            embeddings=FailingEmbeddings(),
        )

        # Should handle error gracefully and return None
        result = await repository._get_embedding_async("Test content")
        assert result is None


@pytest.mark.asyncio
async def test_statistics_gathering():
    """Test repository statistics gathering."""
    with patch(
        "gpt_researcher.persistence.repository.create_async_engine"
    ) as mock_engine:
        mock_engine_instance = AsyncMock()
        mock_engine.return_value = mock_engine_instance

        # Mock async_sessionmaker
        mock_session = AsyncMock()

        def mock_session_factory():
            return mock_session

        with patch(
            "gpt_researcher.persistence.repository.async_sessionmaker"
        ) as mock_sessionmaker:
            mock_sessionmaker.return_value = mock_session_factory

            repository = ScraperRepository(
                db_url="postgresql+asyncpg://test:test@localhost/test"
            )

            # Mock session and results
            mock_session.scalar = AsyncMock(
                side_effect=[100, 75]
            )  # total, with_embeddings
            mock_result = AsyncMock()
            mock_result.one.return_value = ("2024-01-01", "2024-01-31")
            mock_session.execute = AsyncMock(return_value=mock_result)

            # Mock the get_session context manager
            async def mock_get_session():
                yield mock_session

            repository.get_session = AsyncMock(side_effect=mock_get_session)
            repository.get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            repository.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

            stats = await repository.get_statistics()

            assert stats["total_documents"] == 100
            assert stats["documents_with_embeddings"] == 75
            assert stats["oldest_document"] == "2024-01-01"
            assert stats["newest_document"] == "2024-01-31"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
