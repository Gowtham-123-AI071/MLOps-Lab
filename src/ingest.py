import pandas as pd
import numpy as np

np.random.seed(42)
n = 2000

df = pd.DataFrame({
    'user_id': np.random.randint(1, 200, n),
    'movie_id': np.random.randint(1, 100, n),
    'rating': np.random.uniform(1, 5, n),
    'timestamp': range(n)
})

df.to_csv('data/raw/ratings.csv', index=False)

print("ratings.csv created")