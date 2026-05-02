"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class RecommendRequest(BaseModel):
    """Request model for /recommend endpoint."""
    
    user_id: int = Field(
        gt=0,
        description="User ID (must be > 0)",
        example=42
    )
    n: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of recommendations (1-50)"
    )
    exclude_unrated: bool = Field(
        default=False,
        description="Exclude already-rated movies"
    )


class RecommendationItem(BaseModel):
    """Single recommendation."""
    
    movie_id: int = Field(..., description="Movie ID", example=10)
    predicted_rating: float = Field(
        ge=0.5,
        le=5.0,
        description="Predicted rating (0.5-5.0)",
        example=4.8
    )
    rank: int = Field(..., description="Rank (1 = best)", example=1)


class RecommendResponse(BaseModel):
    """Response model for /recommend endpoint."""
    
    user_id: int
    recommendations: List[RecommendationItem]
    timestamp: str
    model_version: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 42,
                "recommendations": [
                    {"movie_id": 10, "predicted_rating": 4.8, "rank": 1},
                    {"movie_id": 15, "predicted_rating": 4.6, "rank": 2}
                ],
                "timestamp": "2026-03-02T10:30:00Z",
                "model_version": "MovieLensRecommender/1"
            }
        }


class PredictionItem(BaseModel):
    """Single (user, movie) pair for prediction."""
    
    user_id: int = Field(gt=0, example=42)
    movie_id: int = Field(gt=0, example=10)


class BatchPredictRequest(BaseModel):
    """Request for batch prediction."""
    
    predictions: List[PredictionItem] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of (user_id, movie_id) pairs (1-100 items)"
    )


class PredictionResult(BaseModel):
    """Single prediction result."""
    
    user_id: int
    movie_id: int
    predicted_rating: float = Field(ge=0.5, le=5.0)


class BatchPredictResponse(BaseModel):
    """Response for batch prediction."""
    
    predictions: List[PredictionResult]
    count: int
    latency_ms: float


class HealthResponse(BaseModel):
    """Response from /health endpoint."""
    
    status: str
    service: str
    version: str
    timestamp: str
    model_version: Optional[str] = None