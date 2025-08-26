"""Persistence layer for GPT Researcher."""

from .models import Base, ScrapedDocument
from .repository import ScraperRepository

__all__ = ["Base", "ScrapedDocument", "ScraperRepository"]
