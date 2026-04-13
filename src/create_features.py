import pandas as pd
import joblib

df = pd.read_csv('data/processed/ratings_clean.csv')

matrix = df.pivot_table(index='user_id', columns='movie_id', values='rating', fill_value=0)

joblib.dump(matrix, 'models/user_similarity.pkl')

print("features created")