
"""
FastAPI server for Crisis Response Environment (OpenEnv compliant)
"""

from fastapi import FastAPI
from openenv.core.env_server.http_server import create_app
import uvicorn

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from models import MyAction, MyObservation
from my_env_environment import MyEnvironment

# Create OpenEnv FastAPI app
app: FastAPI = create_app(
    MyEnvironment,
    MyAction,
    MyObservation,
    env_name="crisis_response_env",
    max_concurrent_envs=5,
)

# Optional root endpoint for browser sanity check
@app.get("/")
def root():
    return {"message": "Crisis Response Env API is running"}

# Browser-friendly GET wrappers for testing (optional)
@app.get("/reset")
def reset_test():
    from fastapi.testclient import TestClient
    client = TestClient(app)
    resp = client.post("/reset")
    return resp.json()

@app.get("/state")
def state_test():
    from fastapi.testclient import TestClient
    client = TestClient(app)
    resp = client.get("/state")
    return resp.json()


def main():
    """
    Entry point for running server locally or via Docker
    """
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,
        reload=True
    )


if __name__ == "__main__":
    main()
    

