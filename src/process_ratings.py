import pandas as pd

df = pd.read_csv('data/raw/ratings.csv')

# remove duplicates
df = df.drop_duplicates()

df.to_csv('data/processed/ratings_clean.csv', index=False)

print("processed done")