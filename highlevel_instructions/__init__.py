"""
Enhanced high-level instruction implementations for GPT Researcher.

This module contains advanced implementations for:
- BatchResearchManager: Enhanced batch research with parallel processing
- ComplexityAnalyzer: Multi-dimensional complexity analysis 
- DualCuratorManager: Enhanced dual curation with quality metrics
"""

from .batch_researcher import BatchResearchManager, ResearchIteration
from .complexity_analyzer import ComplexityAnalyzer, ComplexityMetrics, ResearchRecommendations, ComplexityDimension
from .dual_curator import DualCuratorManager, SourceMetadata, CurationStrategy

__all__ = [
    'BatchResearchManager',
    'ResearchIteration', 
    'ComplexityAnalyzer',
    'ComplexityMetrics',
    'ResearchRecommendations',
    'ComplexityDimension',
    'DualCuratorManager',
    'SourceMetadata',
    'CurationStrategy'
]

__version__ = '1.0.0'