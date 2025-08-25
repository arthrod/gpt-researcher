"""Repository pattern for scraper data persistence with proper async handling."""

import asyncio
import logging
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy.ext.asyncio import (
    AsyncSession, 
    AsyncEngine, 
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy import select, update, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.pool import NullPool, AsyncAdaptedQueuePool

from .models import Base, ScrapedDocument

logger = logging.getLogger(__name__)


class ScraperRepository:
    """Repository for managing scraped document persistence with optimized async operations."""
    
    def __init__(
        self, 
        db_url: str, 
        embeddings: Optional[Any] = None,
        pool_size: int = 20,
        max_overflow: int = 10,
        pool_recycle: int = 3600,
        echo: bool = False
    ):
        """Initialize the repository with database connection and embedding model.
        
        Args:
            db_url: PostgreSQL connection string (must use asyncpg driver)
            embeddings: Embedding model with embed_documents method
            pool_size: Number of connections to maintain in pool
            max_overflow: Maximum overflow connections above pool_size
            pool_recycle: Recycle connections after this many seconds
            echo: Whether to log SQL statements
        """
        self.embeddings = embeddings
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Create engine with async-compatible connection pooling
        self.engine: AsyncEngine = create_async_engine(
            db_url,
            echo=echo,
            poolclass=AsyncAdaptedQueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_recycle=pool_recycle,
            pool_pre_ping=True,  # Test connections before using
        )
        
        # Create async session factory
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def initialize_database(self) -> None:
        """Create database tables if they don't exist."""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    @asynccontextmanager
    async def get_session(self):
        """Context manager for database sessions with proper error handling."""
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    async def _get_embedding_async(self, content: str) -> Optional[List[float]]:
        """Run blocking embedding generation in thread pool to avoid blocking event loop.
        
        Args:
            content: Text content to embed
            
        Returns:
            List of embedding floats or None if no embedding model
        """
        if not self.embeddings:
            return None
            
        loop = asyncio.get_running_loop()
        try:
            # Run the blocking call in a thread pool
            embeddings_list = await loop.run_in_executor(
                self._executor,
                self.embeddings.embed_documents,
                [content]
            )
            return embeddings_list[0] if embeddings_list else None
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    async def save_records(
        self, 
        records: List[Dict[str, Any]], 
        batch_size: int = 50,
        upsert: bool = True
    ) -> int:
        """Save multiple scraped records with batching and upsert support.
        
        Args:
            records: List of dicts with 'url', 'raw_content', and optional 'title'
            batch_size: Number of records to process in each batch
            upsert: Whether to update existing records or skip them
            
        Returns:
            Number of records saved/updated
        """
        saved_count = 0
        
        async with self.get_session() as session:
            # Process in batches for better performance
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                
                # Generate embeddings concurrently for the batch
                embedding_tasks = [
                    self._get_embedding_async(record["raw_content"])
                    for record in batch
                ]
                embeddings = await asyncio.gather(*embedding_tasks)
                
                # Prepare documents for insertion
                documents = []
                for record, embedding in zip(batch, embeddings):
                    doc_data = {
                        "url": record["url"],
                        "title": record.get("title"),
                        "content": record["raw_content"],
                        "embedding": embedding
                    }
                    documents.append(doc_data)
                
                if upsert:
                    # Use PostgreSQL's ON CONFLICT for upsert
                    stmt = insert(ScrapedDocument).values(documents)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["url"],
                        set_={
                            "title": stmt.excluded.title,
                            "content": stmt.excluded.content,
                            "embedding": stmt.excluded.embedding,
                            "updated_at": func.now()
                        }
                    )
                    await session.execute(stmt)
                else:
                    # Regular insert, skip duplicates
                    for doc_data in documents:
                        try:
                            session.add(ScrapedDocument(**doc_data))
                            saved_count += 1
                        except Exception as e:
                            logger.debug(f"Skipping duplicate URL {doc_data['url']}: {e}")
                
                await session.commit()
                logger.info(f"Processed batch of {len(batch)} records")
        
        return saved_count
    
    async def get_document_by_url(self, url: str) -> Optional[ScrapedDocument]:
        """Retrieve a document by its URL.
        
        Args:
            url: The URL to search for
            
        Returns:
            ScrapedDocument or None if not found
        """
        async with self.get_session() as session:
            stmt = select(ScrapedDocument).where(ScrapedDocument.url == url)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()
    
    async def get_documents_batch(
        self, 
        urls: List[str]
    ) -> List[ScrapedDocument]:
        """Retrieve multiple documents by their URLs.
        
        Args:
            urls: List of URLs to retrieve
            
        Returns:
            List of ScrapedDocument objects
        """
        async with self.get_session() as session:
            stmt = select(ScrapedDocument).where(ScrapedDocument.url.in_(urls))
            result = await session.execute(stmt)
            return list(result.scalars().all())
    
    async def update_embeddings(
        self, 
        urls: List[str], 
        batch_size: int = 20
    ) -> int:
        """Update embeddings for existing documents.
        
        Args:
            urls: URLs of documents to update embeddings for
            batch_size: Number of documents to process at once
            
        Returns:
            Number of documents updated
        """
        if not self.embeddings:
            logger.warning("No embedding model configured")
            return 0
        
        updated_count = 0
        
        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i:i + batch_size]
            
            async with self.get_session() as session:
                # Fetch documents
                stmt = select(ScrapedDocument).where(
                    ScrapedDocument.url.in_(batch_urls)
                )
                result = await session.execute(stmt)
                documents = result.scalars().all()
                
                # Generate embeddings concurrently
                embedding_tasks = [
                    self._get_embedding_async(doc.content)
                    for doc in documents
                ]
                embeddings = await asyncio.gather(*embedding_tasks)
                
                # Update documents
                for doc, embedding in zip(documents, embeddings):
                    if embedding:
                        doc.embedding = embedding
                        updated_count += 1
                
                await session.commit()
                logger.info(f"Updated embeddings for {len(documents)} documents")
        
        return updated_count
    
    async def search_by_embedding(
        self, 
        query_embedding: List[float], 
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[ScrapedDocument]:
        """Search documents by embedding similarity using pgvector.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar documents
        """
        # This would require pgvector extension for PostgreSQL
        # Implementation depends on pgvector being installed
        async with self.get_session() as session:
            # Placeholder for pgvector similarity search
            # Would use something like:
            # stmt = select(ScrapedDocument).order_by(
            #     ScrapedDocument.embedding.cosine_distance(query_embedding)
            # ).limit(limit)
            pass
    
    async def cleanup_old_documents(self, days: int = 30) -> int:
        """Remove documents older than specified days.
        
        Args:
            days: Number of days to keep documents
            
        Returns:
            Number of documents deleted
        """
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        async with self.get_session() as session:
            stmt = select(ScrapedDocument).where(
                ScrapedDocument.created_at < cutoff_date
            )
            result = await session.execute(stmt)
            old_docs = result.scalars().all()
            
            for doc in old_docs:
                await session.delete(doc)
            
            await session.commit()
            logger.info(f"Deleted {len(old_docs)} old documents")
            return len(old_docs)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics.
        
        Returns:
            Dictionary with statistics
        """
        from sqlalchemy import func
        
        async with self.get_session() as session:
            # Count total documents
            count_stmt = select(func.count()).select_from(ScrapedDocument)
            total_count = await session.scalar(count_stmt)
            
            # Count documents with embeddings
            embedding_stmt = select(func.count()).select_from(ScrapedDocument).where(
                ScrapedDocument.embedding.isnot(None)
            )
            embedding_count = await session.scalar(embedding_stmt)
            
            # Get date range
            date_stmt = select(
                func.min(ScrapedDocument.created_at),
                func.max(ScrapedDocument.created_at)
            )
            result = await session.execute(date_stmt)
            min_date, max_date = result.one()
            
            return {
                "total_documents": total_count or 0,
                "documents_with_embeddings": embedding_count or 0,
                "oldest_document": min_date,
                "newest_document": max_date
            }
    
    async def close(self) -> None:
        """Close database connections and cleanup resources."""
        await self.engine.dispose()
        self._executor.shutdown(wait=True)
        logger.info("Repository closed successfully")