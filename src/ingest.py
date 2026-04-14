import pandas as pd
import json
import logging
from pathlib import Path
from src.config import RATINGS_SCHEMA, DATA_PATHS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RatingsLoader:

    def __init__(self, filepath=DATA_PATHS['raw']):
        self.filepath = filepath
        self.raw_df = None
        self.clean_df = None

    def load(self):
        logger.info(f"Loading data from {self.filepath}")
        self.raw_df = pd.read_csv(self.filepath, sep='\t')
        logger.info(f"Loaded {len(self.raw_df)} rows")
        return self.raw_df

    def validate(self, df):
        logger.info("Validating data...")
        
        # Remove invalid ratings
        df = df[(df['rating'] >= 0.5) & (df['rating'] <= 5.0)]
        
        # Remove invalid user/movie ids
        df = df[(df['user_id'] > 0) & (df['movie_id'] > 0)]
        
        logger.info(f"Remaining rows after validation: {len(df)}")
        return df

    def save(self, df):
        Path(DATA_PATHS['processed']).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATHS['processed'], index=False)
        logger.info("Saved clean data")

    def save_report(self, df):
        report = {
            "rows": len(df),
            "columns": list(df.columns)
        }
        Path(DATA_PATHS['validation_report']).parent.mkdir(parents=True, exist_ok=True)
        
        with open(DATA_PATHS['validation_report'], 'w') as f:
            json.dump(report, f, indent=2)

        logger.info("Saved validation report")


def main():
    loader = RatingsLoader()
    df = loader.load()
    clean_df = loader.validate(df)
    loader.save(clean_df)
    loader.save_report(clean_df)


if __name__ == "__main__":
    main()