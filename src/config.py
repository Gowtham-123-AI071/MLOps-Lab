"""Configuration and schema for MovieLens data pipeline."""

RATINGS_SCHEMA = {
    'user_id': {
        'dtype': 'int64',
        'min': 1,
        'max': 1000,
        'nullable': False
    },
    'movie_id': {
        'dtype': 'int64',
        'min': 1,
        'max': 1700,
        'nullable': False
    },
    'rating': {
        'dtype': 'float64',
        'min': 0.5,
        'max': 5.0,
        'nullable': False
    },
    'timestamp': {
        'dtype': 'int64',
        'min': 1000000000,
        'max': 2000000000,
        'nullable': False
    }
}

DATA_PATHS = {
    'raw': 'data/raw/ratings.csv',
    'processed': 'data/processed/ratings_clean.csv',
    'validation_report': 'evaluations/validation_report.json'
}