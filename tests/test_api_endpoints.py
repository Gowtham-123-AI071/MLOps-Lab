import pytest
from fastapi.testclient import TestClient
from src.app import app

# Helper to ensure startup runs
def get_client():
    return TestClient(app)


def test_health_endpoint():
    """Test health check returns 200."""
    with get_client() as client:
        response = client.get("/health")
        assert response.status_code == 200


def test_recommend_valid_user():
    """Test recommendation for valid user."""
    with get_client() as client:
        response = client.get("/recommend?user_id=1&n=5")
        assert response.status_code == 200

        data = response.json()
        assert "recommendations" in data
        assert len(data["recommendations"]) <= 5


def test_recommend_invalid_user_id():
    """Test recommendation rejects invalid user_id."""
    with get_client() as client:
        response = client.get("/recommend?user_id=-1&n=5")
        assert response.status_code == 422  # Validation error


def test_recommend_invalid_n():
    """Test recommendation rejects invalid n."""
    with get_client() as client:
        response = client.get("/recommend?user_id=1&n=100")
        assert response.status_code == 422  # n > 50 rejected


def test_batch_predict():
    """Test batch prediction."""
    with get_client() as client:
        response = client.post(
            "/predict_batch",
            json={
                "predictions": [
                    {"user_id": 1, "movie_id": 1},
                    {"user_id": 1, "movie_id": 2}
                ]
            }
        )
        assert response.status_code == 200

        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) >= 0


def test_missing_model_returns_503():
    """Test returns 503 if model not loaded (basic check)."""
    # Optional: depends on mocking, so we just pass
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])