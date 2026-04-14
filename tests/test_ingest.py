import pytest
from src.ingest import RatingsLoader

def test_load():
    loader = RatingsLoader()
    df = loader.load()
    assert df is not None
    assert len(df) > 0

def test_validate():
    loader = RatingsLoader()
    df = loader.load()
    clean_df = loader.validate(df)
    assert clean_df is not None
    assert len(clean_df) <= len(df)