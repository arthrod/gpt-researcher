"""Persistence layer for GPT Researcher."""

from .repository import ScraperRepository
from .models import Base, ScrapedDocument

__all__ = ["ScraperRepository", "Base", "ScrapedDocument"]