"""FastAPI application for MovieLens recommendations."""

import logging
import time
import joblib
import os
from datetime import datetime
from typing import List
import numpy as np
from fastapi import Query
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.schemas import (
    RecommendRequest,
    RecommendResponse,
    RecommendationItem,
    BatchPredictRequest,
    BatchPredictResponse,
    PredictionResult,
    HealthResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MovieLens Recommender API",
    description="K-NN based recommendation system",
    version="1.0.0"
)

# CORS middleware (allows requests from other domains)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify actual domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and features (loaded on startup)
model = None
features = None
model_version = "MovieLensRecommender/1"


@app.on_event("startup")
async def startup_event():
    """Load model and features on startup."""
    global model, features
    
    try:
        logger.info("Loading model...")
        model_path = os.getenv("MODEL_PATH", "models/model.pkl")
        model = joblib.load(model_path)
        logger.info(f"✓ Model loaded from {model_path}")
        
        logger.info("Loading features...")
        features_path = os.getenv("FEATURES_PATH", "models/rating_features.pkl")
        features = joblib.load(features_path)
        logger.info(f"✓ Features loaded from {features_path}")
        
    except FileNotFoundError as e:
        logger.error(f"✗ Model file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        raise


model = None

model = None
features = None
model_version = "MovieLensRecommender/1"

@app.on_event("startup")
def load_resources():
    global model, features

    try:
        model = joblib.load("models/model.pkl")
        features = joblib.load("models/rating_features.pkl")

        print("✓ Model and features loaded")

    except Exception as e:
        print(f"Error loading resources: {e}")
        model = None
        features = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with service status
    
    Status codes:
        - 200: Service healthy
        - 503: Model not loaded
    """
    if model is None or features is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return HealthResponse(
        status="healthy",
        service="MovieLens Recommender API",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat() + "Z",
        model_version=model_version
    )


from datetime import datetime

from fastapi import Query
from datetime import datetime

@app.get("/recommend")
def recommend(
    user_id: int = Query(..., gt=0),
    n: int = Query(5, gt=0, le=50)   # ✅ FINAL FIX
):
    try:
        movie_ids = model.features.movie_ids

        predictions = []

        for movie_id in movie_ids:
            rating = model.predict_rating(user_id, movie_id)
            predictions.append((movie_id, rating))

        predictions.sort(key=lambda x: x[1], reverse=True)

        recommendations = []
        for rank, (movie_id, rating) in enumerate(predictions[:n], start=1):
            recommendations.append({
                "movie_id": int(movie_id),
                "predicted_rating": round(float(rating), 2),
                "rank": rank
            })

        return {
            "user_id": int(user_id),
            "recommendations": recommendations,
            "timestamp": datetime.utcnow().isoformat(),
            "model_version": "MovieLensRecommender/1"
        }

    except Exception as e:
        return {"detail": f"Recommendation generation failed: {str(e)}"}
    """
    Get N movie recommendations for a user.
    
    Args:
        user_id: Target user ID
        n: Number of recommendations (1-50)
    
    Returns:
        RecommendResponse with ranked recommendations
    
    Raises:
        422: Invalid input (user_id <= 0, n out of range)
        404: User not in training data
        500: Model inference failed
    
    Example request:
        GET /recommend?user_id=42&n=5
    
    Example response:
        {
            "user_id": 42,
            "recommendations": [
                {"movie_id": 10, "predicted_rating": 4.8, "rank": 1},
                {"movie_id": 15, "predicted_rating": 4.6, "rank": 2}
            ],
            "timestamp": "2026-03-02T10:30:00Z",
            "model_version": "MovieLensRecommender/1"
        }
    """
    start_time = time.time()
    
    try:
        # Validate user exists in training data
        if user_id not in features.user_ids:
            logger.warning(f"User {user_id} not in training data")
            raise HTTPException(
                status_code=404,
                detail=f"User {user_id} not found in model training data"
            )
        
        # Get user's rating vector
        user_idx = np.where(features.user_ids == user_id)[0][0]
        user_ratings = features.ratings_matrix.iloc[user_idx].values
        
        # Find K nearest neighbors
        X = features.ratings_matrix.values
        distances, indices = model.kneighbors(
            [user_ratings],
            n_neighbors=min(5, len(X))
        )
        
        # Get all movie ratings from neighbors
        neighbor_ratings = features.ratings_matrix.iloc[indices[0]].mean(axis=0)
        
        # Get movies user hasn't rated
        unrated_mask = user_ratings == 0
        unrated_movies = np.where(unrated_mask)[0]
        
        if len(unrated_movies) == 0:
            logger.warning(f"User {user_id} has rated all movies")
            raise HTTPException(
                status_code=400,
                detail=f"User {user_id} has already rated all movies"
            )
        
        # Get predictions for unrated movies
        movie_predictions = {}
        for movie_idx in unrated_movies:
            movie_predictions[features.movie_ids[movie_idx]] = neighbor_ratings[movie_idx]
        
        # Filter predictionsby threshold
        predictions = [
            (movie_id, rating) for movie_id, rating in movie_predictions.items()
            if rating >= 0.5  # Minimum rating threshold
        ]
        
        # Sort by rating descending
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Take top N
        top_predictions = predictions[:n]
        
        # Build response
        recommendations = [
            RecommendationItem(
                movie_id=int(movie_id),
                predicted_rating=float(rating),
                rank=rank + 1
            )
            for rank, (movie_id, rating) in enumerate(top_predictions)
        ]
        
        logger.info(f"Generated {len(recommendations)} recommendations for user {user_id} ({time.time() - start_time:.3f}s)")
        
        return RecommendResponse(
            user_id=user_id,
            recommendations=recommendations,
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_version=model_version
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate recommendations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation generation failed: {str(e)}"
        )


@app.post("/predict_batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):

    start_time = time.time()

    try:
        if model is None or features is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        results = []

        for pred in request.predictions:
            user_id = pred.user_id
            movie_id = pred.movie_id

            # Skip invalid users/movies
            if user_id not in features.user_ids:
                continue
            if movie_id not in features.movie_ids:
                continue

            # ✅ CORRECT: use your model method
            predicted_rating = float(
                model.predict_rating(user_id, movie_id)
            )

            results.append(PredictionResult(
                user_id=int(user_id),
                movie_id=int(movie_id),
                predicted_rating=predicted_rating
            ))

        elapsed = time.time() - start_time

        return BatchPredictResponse(
            predictions=results,
            count=len(results),
            latency_ms=elapsed * 1000
        )

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/similar_users/{user_id}")
async def similar_users(
    user_id: int,
    k: int = Query(5, ge=1, le=20, description="Number of similar users")
):
    """
    Find K most similar users to target user.
    
    Args:
        user_id: Target user ID
        k: Number of similar users to return
    
    Returns:
        JSON with list of similar users and similarity scores
    """
    try:
        if user_id not in features.user_ids:
            raise HTTPException(
                status_code=404,
                detail=f"User {user_id} not found"
            )
        
        similar = features.get_similar_users(user_id, n=k)
        
        return {
            "user_id": user_id,
            "similar_users": [
                {"user_id": int(uid), "similarity": float(sim)}
                for uid, sim in similar
            ]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get similar users: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4
    )