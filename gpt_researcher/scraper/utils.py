import hashlib
import logging
import re

from urllib.parse import parse_qs, urljoin, urlparse

import bs4

from bs4 import BeautifulSoup

try:
    from sqlalchemy import Integer, String, Text
    from sqlalchemy.dialects.postgresql import insert
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

    HAS_SQLA = True
    HAS_PG = True
except Exception:
    HAS_SQLA = False
    HAS_PG = False

# SQLAlchemy models for persisting scraped data (optional)
if HAS_SQLA:

    class Base(DeclarativeBase):
        pass

    class ScrapedDocument(Base):
        __tablename__ = "scraped_documents"
        id: "Mapped[int]" = mapped_column(Integer, primary_key=True, autoincrement=True)
        url: "Mapped[str]" = mapped_column(String, unique=True)
        title: "Mapped[str | None]" = mapped_column(String, nullable=True)
        content: "Mapped[str]" = mapped_column(Text)
        # Store embeddings as JSON/text to avoid dialect-specific ARRAY
        embedding: "Mapped[str | None]" = mapped_column(Text, nullable=True)


def get_relevant_images(soup: BeautifulSoup, url: str) -> list:
    """Extract relevant images from the page"""
    image_urls = []

    try:
        # Find all img tags with src attribute
        all_images = soup.find_all("img", src=True)

        for img in all_images:
            img_src = urljoin(url, img["src"])
            if img_src.startswith(("http://", "https://")):
                score = 0
                # Check for relevant classes
                if any(
                    cls in img.get("class", [])
                    for cls in [
                        "header",
                        "featured",
                        "hero",
                        "thumbnail",
                        "main",
                        "content",
                    ]
                ):
                    score = 4  # Higher score
                # Check for size attributes
                elif img.get("width") and img.get("height"):
                    width = parse_dimension(img["width"])
                    height = parse_dimension(img["height"])
                    if width and height:
                        if width >= 2000 and height >= 1000:
                            score = 3  # Medium score (very large images)
                        elif width >= 1600 or height >= 800:
                            score = 2  # Lower score
                        elif width >= 800 or height >= 500:
                            score = 1  # Lowest score
                        elif width >= 500 or height >= 300:
                            score = 0  # Lowest score
                        else:
                            continue  # Skip small images

                image_urls.append({"url": img_src, "score": score})

        # Sort images by score (highest first)
        sorted_images = sorted(image_urls, key=lambda x: x["score"], reverse=True)

        return sorted_images[:10]  # Ensure we don't return more than 10 images in total

    except Exception as e:
        logging.error(f"Error in get_relevant_images: {e}")
        return []


def parse_dimension(value: str) -> int:
    """Parse dimension value, handling px units"""
    if value.lower().endswith("px"):
        value = value[:-2]  # Remove 'px' suffix
    try:
        return int(value)  # Convert to float first to handle decimal values
    except ValueError as e:
        print(f"Error parsing dimension value {value}: {e}")
        return None


def extract_title(soup: BeautifulSoup) -> str:
    """Extract the title from the BeautifulSoup object"""
    return soup.title.string if soup.title else ""


def get_image_hash(image_url: str) -> str:
    """Calculate a simple hash based on the image filename and essential query parameters"""
    try:
        parsed_url = urlparse(image_url)

        # Extract the filename
        filename = parsed_url.path.split("/")[-1]

        # Extract essential query parameters (e.g., 'url' for CDN-served images)
        query_params = parse_qs(parsed_url.query)
        essential_params = query_params.get("url", [])

        # Combine filename and essential parameters
        image_identifier = filename + "".join(essential_params)

        # Calculate hash
        return hashlib.md5(image_identifier.encode()).hexdigest()
    except Exception as e:
        logging.error(f"Error calculating image hash for {image_url}: {e}")
        return None


def clean_soup(soup: BeautifulSoup) -> BeautifulSoup:
    """Clean the soup by removing unwanted tags"""
    for tag in soup.find_all([
        "script",
        "style",
        "footer",
        "header",
        "nav",
        "menu",
        "sidebar",
        "svg",
    ]):
        tag.decompose()

    disallowed_class_set = {"nav", "menu", "sidebar", "footer"}

    # clean tags with certain classes
    def does_tag_have_disallowed_class(elem) -> bool:
        if not isinstance(elem, bs4.Tag):
            return False

        return any(
            cls_name in disallowed_class_set for cls_name in elem.get("class", [])
        )

    for tag in soup.find_all(does_tag_have_disallowed_class):
        tag.decompose()

    return soup


def get_text_from_soup(soup: BeautifulSoup) -> str:
    """Get the relevant text from the soup with improved filtering"""
    text = soup.get_text(strip=True, separator="\n")
    # Remove excess whitespace
    text = re.sub(r"\s{2,}", " ", text)
    return text


async def save_scraped_data(records: list[dict], db_url: str, embeddings) -> None:
    """Persist scraped data to Postgres with embeddings.

    Args:
        records: List of scraped records with ``url``, ``title`` and ``raw_content`` keys.
        db_url: SQLAlchemy connection string using the ``psycopg`` async driver.
        embeddings: Embedding function implementing ``embed_documents``.
    """
    engine = create_async_engine(db_url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSession(engine) as session:
        # Prepare raw contents for batch embedding
        contents = [r.get("raw_content", "") for r in records]

        # Batch embed off the event loop to avoid blocking
        if embeddings and contents:
            try:
                import asyncio

                emb_list = await asyncio.to_thread(embeddings.embed_documents, contents)
            except Exception as e:
                logging.error("Embedding failed; storing without embeddings: %s", e)
                emb_list = [[] for _ in contents]
        else:
            emb_list = [[] for _ in contents]

        # Upsert each record by URL
        for rec, emb in zip(records, emb_list, strict=False):
            stmt = (
                insert(ScrapedDocument)
                .values(
                    url=rec["url"],
                    title=rec.get("title"),
                    content=rec.get("raw_content", ""),
                    embedding=emb,
                )
                .on_conflict_do_update(
                    index_elements=[ScrapedDocument.url],
                    set_={
                        "title": rec.get("title"),
                        "content": rec.get("raw_content", ""),
                        "embedding": emb,
                    },
                )
            )
            await session.execute(stmt)

        await session.commit()

    await engine.dispose()
