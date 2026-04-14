import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

df = pd.DataFrame({
    'user_id': np.random.randint(1, 200, 2000),
    'movie_id': np.random.randint(1, 100, 2000),
    'rating': np.random.uniform(1, 5, 2000),
    'timestamp': np.random.randint(1000000000, 1500000000, 2000)
})

Path('data/raw').mkdir(parents=True, exist_ok=True)
df.to_csv('data/raw/ratings.csv', sep='\t', index=False)

print("Data generated successfully")
