import subprocess
import pytest
import time
import requests

def test_dockerfile_builds():
    """Test Dockerfile builds without errors."""
    result = subprocess.run(
        ["docker", "build", "-t", "movielens-api:test", "."],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Build failed: {result.stderr}"


def test_dockerfile_prod_builds():
    """Test production Dockerfile builds."""
    result = subprocess.run(
        ["docker", "build", "-f", "Dockerfile.prod", "-t", "movielens-api:test-prod", "."],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Prod build failed: {result.stderr}"


def test_image_size_under_limit():
    """Test production image size < 500MB (actual size)."""
    import json

    result = subprocess.run(
        ["docker", "inspect", "movielens-api:test-prod"],
        capture_output=True,
        text=True
    )

    data = json.loads(result.stdout)

    size_bytes = data[0]["Size"]
    size_mb = size_bytes / (1024 * 1024)

    assert size_mb < 500, f"Image too large: {size_mb:.2f}MB > 500MB"


def test_healthcheck_works():
    """Test Docker healthcheck passes."""
    # Start container
    result = subprocess.run(
        ["docker", "run", "-d", "-p", "9000:8000", "movielens-api:test"],
        capture_output=True,
        text=True
    )
    
    container_id = result.stdout.strip()
    
    try:
        # Wait for container to start
        time.sleep(5)
        
        # Test health
        response = requests.get("http://localhost:9000/health")
        assert response.status_code == 200
        
    finally:
        # Stop container
        subprocess.run(["docker", "stop", container_id], capture_output=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])