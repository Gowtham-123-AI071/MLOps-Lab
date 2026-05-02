import pandas as pd
import numpy as np
from src.features import RatingFeatures
from sklearn.model_selection import train_test_split
from src.sweep_experiments import run_parameter_sweep, identify_best_run

# Load data
ratings_df = pd.read_csv('data/processed/ratings_clean.csv')

# Load features
features = RatingFeatures.load('models/rating_features.pkl')

# Target (average rating per user)
y = ratings_df.groupby('user_id')['rating'].mean()

# Split users
train_users, test_users = train_test_split(
    features.user_ids, test_size=0.2, random_state=42
)

# Create train/test sets
X_train = features.ratings_matrix.loc[train_users].values
X_test = features.ratings_matrix.loc[test_users].values

y_train = pd.DataFrame({'rating': y.loc[train_users].values})
y_test = pd.DataFrame({'rating': y.loc[test_users].values})

# Run sweep
k_values = [3, 5, 10, 15, 20]

results = run_parameter_sweep(
    k_values=k_values,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    experiment_name="movielens_knn_sweep_v2"
)

# Best model
best_k, best_result = identify_best_run(results, metric="rmse")

print("\n" + "="*50)
print(f"Best K: {best_k}")
print(f"RMSE: {best_result['rmse']:.3f}")
print(f"MAE: {best_result['mae']:.3f}")
print("="*50)