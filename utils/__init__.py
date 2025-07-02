"""
YABATECH Course Recommendation System - Utils Package

This package contains utility modules for data preprocessing and recommendation engine.
"""

from .data_preprocessor import DataPreprocessor, validate_input_data, format_user_input
from .recommendation_engine import RecommendationEngine

__version__ = "1.0.0"
__author__ = "YABATECH Development Team"

__all__ = [
    'DataPreprocessor',
    'RecommendationEngine',
    'validate_input_data',
    'format_user_input'
]