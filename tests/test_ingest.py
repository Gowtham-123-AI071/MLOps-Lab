import pytest
import pandas as pd
from src.ingest import RatingsLoader, RatingsValidator

def test_load():
    loader = RatingsLoader()
    df = loader.load()
    assert len(df) > 0

def test_deduplication():
    loader = RatingsLoader()
    df = loader.load()
    df2 = loader.deduplicate(df)
    assert len(df2) <= len(df)

def test_validate_columns():
    validator = RatingsValidator()
    df = pd.DataFrame({
        'user_id': [1],
        'movie_id': [1],
        'rating': [4],
        'timestamp': [1234567890]
    })
    assert validator.validate_columns(df) == True

def test_validate_types():
    validator = RatingsValidator()
    df = pd.DataFrame({
        'user_id': ['a'],
        'movie_id': [1],
        'rating': [4],
        'timestamp': [123]
    })
    df, errors = validator.validate_types(df)
    assert errors >= 1

def test_validate_ranges():
    validator = RatingsValidator()
    df = pd.DataFrame({
        'user_id': [1],
        'movie_id': [1],
        'rating': [10],  # invalid
        'timestamp': [1234567890]
    })
    df, removed = validator.validate_ranges(df)
    assert removed >= 1

def test_validate_nulls():
    validator = RatingsValidator()
    df = pd.DataFrame({
        'user_id': [1],
        'movie_id': [None],
        'rating': [4],
        'timestamp': [1234567890]
    })
    df, removed = validator.validate_nulls(df)
    assert removed >= 1