"""
FastAPI server for Crisis Response Product + OpenEnv compatibility
"""

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import sys
from pathlib import Path

# Add server folder to path
sys.path.append(str(Path(__file__).resolve().parent))

from models import MyAction, MyObservation
from my_env_environment import MyEnvironment


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Crisis Response AI API",
    description="Backend API for AI-powered crisis response simulation",
    version="1.0.0"
)

# Single environment instance for product demo
env = MyEnvironment()


# -----------------------------
# Request Models
# -----------------------------
class IncidentInput(BaseModel):
    incident_type: str
    severity: str
    location: str
    people_affected: int


class ActionInput(BaseModel):
    action_type: str
    incident_id: str | None = None
    resource_type: str | None = None
    amount: int = 0
    priority: int | None = None


# -----------------------------
# Basic Routes
# -----------------------------
@app.get("/")
def root():
    return {
        "message": "Crisis Response AI backend is running",
        "docs": "/docs"
    }


@app.post("/reset")
def reset_environment():
    """
    Reset simulation environment
    """
    observation = env.reset()
    return observation.model_dump()


@app.get("/state")
def get_state():
    """
    Get current environment state
    """
    observation = env._build_observation(reward=0.0, done=False)
    return observation.model_dump()


# -----------------------------
# Product Routes
# -----------------------------
@app.post("/incident")
def create_incident(data: IncidentInput):
    """
    Add custom incident from frontend/user input
    """
    observation = env.add_incident(
        incident_type=data.incident_type,
        severity=data.severity,
        location=data.location,
        people_affected=data.people_affected
    )

    return {
        "message": "Incident created successfully",
        "state": observation.model_dump()
    }


@app.post("/action")
def apply_action(data: ActionInput):
    """
    Apply action selected by AI/user
    """
    action = MyAction(
        action_type=data.action_type,
        incident_id=data.incident_id,
        resource_type=data.resource_type,
        amount=data.amount,
        priority=data.priority
    )

    observation = env.step(action)

    return {
        "message": "Action applied successfully",
        "state": observation.model_dump()
    }


@app.post("/agent/decide")
def agent_decide():
    """
    Temporary rule-based agent.
    Later we will replace this with LLM agent.
    """
    observation = env._build_observation(reward=0.0, done=False)

    unresolved = [
        inc for inc in observation.active_incidents
        if not inc.resolved
    ]

    if unresolved:
        action = {
            "action_type": "resolve_incident",
            "incident_id": unresolved[0].incident_id,
            "resource_type": None,
            "amount": 0,
            "priority": None
        }
    else:
        action = {
            "action_type": "do_nothing",
            "incident_id": None,
            "resource_type": None,
            "amount": 0,
            "priority": None
        }

    return {
        "message": "Agent decision generated",
        "observation": observation.model_dump(),
        "action": action
    }


@app.post("/agent/act")
def agent_act():
    """
    Agent decides and immediately applies action.
    Useful for frontend demo button.
    """
    decision = agent_decide()
    action_data = decision["action"]

    action = MyAction(**action_data)
    observation = env.step(action)

    return {
        "message": "Agent action applied",
        "action": action_data,
        "state": observation.model_dump()
    }


# -----------------------------
# Run server
# -----------------------------
def main():
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,
        reload=True
    )


if __name__ == "__main__":
    main()