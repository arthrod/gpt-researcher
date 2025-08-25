"""Refactored scraper utilities using the repository pattern."""

from typing import List, Dict, Any, Optional
import logging
from ..persistence import ScraperRepository

logger = logging.getLogger(__name__)


async def save_scraped_data_optimized(
    records: List[Dict[str, Any]], 
    db_url: str, 
    embeddings: Optional[Any] = None,
    batch_size: int = 50,
    upsert: bool = True
) -> None:
    """Optimized version of save_scraped_data using repository pattern.
    
    This function properly handles async operations and uses batching for better performance.
    
    Args:
        records: List of scraped records with 'url', 'title' and 'raw_content' keys
        db_url: PostgreSQL connection string (should use asyncpg driver)
        embeddings: Embedding model implementing embed_documents method
        batch_size: Number of records to process in each batch
        upsert: Whether to update existing records or skip them
    """
    if not records:
        logger.warning("No records to save")
        return
    
    # Initialize repository with connection pooling
    repository = ScraperRepository(
        db_url=db_url,
        embeddings=embeddings,
        pool_size=20,
        max_overflow=10
    )
    
    try:
        # Initialize database tables if needed
        await repository.initialize_database()
        
        # Save records with proper batching and async handling
        saved_count = await repository.save_records(
            records=records,
            batch_size=batch_size,
            upsert=upsert
        )
        
        logger.info(f"Successfully saved {saved_count} records to database")
        
        # Get statistics
        stats = await repository.get_statistics()
        logger.info(f"Database statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Error saving scraped data: {e}")
        raise
    finally:
        # Always close the repository to clean up resources
        await repository.close()


class ScraperDataManager:
    """Manager class for handling scraped data with persistent repository."""
    
    def __init__(
        self, 
        db_url: str, 
        embeddings: Optional[Any] = None,
        pool_size: int = 20
    ):
        """Initialize the scraper data manager.
        
        Args:
            db_url: PostgreSQL connection string
            embeddings: Embedding model
            pool_size: Database connection pool size
        """
        self.repository = ScraperRepository(
            db_url=db_url,
            embeddings=embeddings,
            pool_size=pool_size
        )
        self._initialized = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def initialize(self) -> None:
        """Initialize the database and repository."""
        if not self._initialized:
            await self.repository.initialize_database()
            self._initialized = True
    
    async def save_batch(
        self, 
        records: List[Dict[str, Any]], 
        batch_size: int = 50
    ) -> int:
        """Save a batch of records.
        
        Args:
            records: Records to save
            batch_size: Batch size for processing
            
        Returns:
            Number of records saved
        """
        return await self.repository.save_records(
            records=records,
            batch_size=batch_size
        )
    
    async def get_existing_urls(self, urls: List[str]) -> List[str]:
        """Check which URLs already exist in the database.
        
        Args:
            urls: URLs to check
            
        Returns:
            List of URLs that already exist
        """
        existing_docs = await self.repository.get_documents_batch(urls)
        return [doc.url for doc in existing_docs]
    
    async def filter_new_urls(self, urls: List[str]) -> List[str]:
        """Filter out URLs that already exist in the database.
        
        Args:
            urls: URLs to filter
            
        Returns:
            List of new URLs not in database
        """
        existing_urls = set(await self.get_existing_urls(urls))
        return [url for url in urls if url not in existing_urls]
    
    async def update_embeddings_for_urls(
        self, 
        urls: List[str], 
        batch_size: int = 20
    ) -> int:
        """Update embeddings for specific URLs.
        
        Args:
            urls: URLs to update embeddings for
            batch_size: Batch size for processing
            
        Returns:
            Number of documents updated
        """
        return await self.repository.update_embeddings(
            urls=urls,
            batch_size=batch_size
        )
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Statistics dictionary
        """
        return await self.repository.get_statistics()
    
    async def cleanup_old_data(self, days: int = 30) -> int:
        """Clean up old scraped data.
        
        Args:
            days: Keep data from last N days
            
        Returns:
            Number of records deleted
        """
        return await self.repository.cleanup_old_documents(days=days)
    
    async def close(self) -> None:
        """Close the repository and clean up resources."""
        await self.repository.close()


# Example usage function
async def example_usage():
    """Example of how to use the optimized scraper data persistence."""
    import os
    from langchain_openai import OpenAIEmbeddings
    
    # Sample scraped data
    records = [
        {
            "url": "https://example.com/page1",
            "title": "Example Page 1",
            "raw_content": "This is the content of page 1..."
        },
        {
            "url": "https://example.com/page2",
            "title": "Example Page 2", 
            "raw_content": "This is the content of page 2..."
        }
    ]
    
    # Database connection string
    db_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/dbname")
    
    # Initialize embeddings (optional)
    embeddings = OpenAIEmbeddings() if os.getenv("OPENAI_API_KEY") else None
    
    # Method 1: Using the simple function
    await save_scraped_data_optimized(
        records=records,
        db_url=db_url,
        embeddings=embeddings,
        batch_size=50,
        upsert=True
    )
    
    # Method 2: Using the manager class for more control
    async with ScraperDataManager(db_url, embeddings) as manager:
        # Check for existing URLs
        urls_to_check = [r["url"] for r in records]
        new_urls = await manager.filter_new_urls(urls_to_check)
        print(f"Found {len(new_urls)} new URLs to process")
        
        # Save only new records
        new_records = [r for r in records if r["url"] in new_urls]
        if new_records:
            saved = await manager.save_batch(new_records)
            print(f"Saved {saved} new records")
        
        # Get statistics
        stats = await manager.get_statistics()
        print(f"Database stats: {stats}")
        
        # Clean up old data (optional)
        deleted = await manager.cleanup_old_data(days=30)
        print(f"Deleted {deleted} old records")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())