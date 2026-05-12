---
title: Crisis Response Coordinator Environment
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
  - real-world
  - crisis-management
  - decision-making
---

# Crisis Response Coordinator Environment

A real-world OpenEnv environment simulating crisis response coordination. The agent must manage emergency incidents by dispatching resources, allocating supplies, requesting backup, broadcasting alerts, prioritizing incidents, and resolving crises to minimize impact on affected people.

## Environment Description

This environment models a crisis response scenario where multiple incidents (fires, floods, medical emergencies) occur simultaneously. The agent acts as a coordinator making decisions to:

- Dispatch available resources to incidents
- Allocate specific resource types
- Request additional backup resources
- Broadcast public alerts to slow crisis spread
- Prioritize high-severity incidents
- Resolve incidents to stop the crisis

The goal is to resolve all incidents as quickly as possible while managing limited resources and system load.

## Action Space

Actions are defined by the `MyAction` model:

- `action_type`: One of ["dispatch_team", "allocate_resource", "request_backup", "broadcast_alert", "prioritize_incident", "resolve_incident", "do_nothing"]
- `incident_id`: Target incident ID (required for prioritize_incident, resolve_incident)
- `resource_type`: Resource type to allocate (required for allocate_resource)
- `amount`: Quantity to allocate
- `priority`: Priority level (1-5)

## Observation Space

Observations are defined by the `MyObservation` model:

- `time_step`: Current simulation step
- `active_incidents`: List of ongoing incidents with id, type, severity, location, people_affected, resolved
- `resources`: List of available resources with type, available count, in_use count
- `total_people_affected`: Total people impacted across all incidents
- `resolved_incidents`: Number of incidents resolved
- `system_load`: Current system load (0.0-1.0)
- `response_efficiency`: Response efficiency metric (0.0-1.0)
- `done`: Whether all incidents are resolved
- `reward`: Step reward (negative for growing crises, positive for resolutions)

## Reward Function

The reward provides partial progress signals:

- **Penalty**: -0.01 per person affected (encourages quick resolution)
- **Resolution Bonus**: +10 per incident resolved in the step
- **Priority Bonus**: +5 per high-severity incident (rewards handling critical cases)
- **Alert Effect**: Broadcasting alerts halves incident growth rates

Episode ends when all incidents are resolved (done=True).

## Tasks

Three difficulty levels with agent graders:

- **Easy**: Classify incident urgency (low/medium/high)
- **Medium**: Allocate resources to incidents
- **Hard**: Multi-incident crisis coordination and prioritization

Graders return scores from 0.0 to 1.0 based on accuracy and partial credit.

## Setup Instructions

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```
HF_TOKEN=your_huggingface_token
```

3. Run the server:
```bash
python -m server.app
```

   Or run the FastAPI server directly:
```bash
cd server
python app.py
```

4. Test the environment:
```python
from client import myEnv

env = myEnv(base_url="http://localhost:7860")
observation = env.reset()
# ... interact with environment
```

### Docker Build

```bash
docker build -t crisis-response-coordinator -f server/Dockerfile .
docker run -p 7860:7860 crisis-response-coordinator
```

### Hugging Face Spaces Deployment

The environment is configured for HF Spaces deployment with the Docker SDK. Push this repository to HF Spaces to deploy automatically.

### Baseline Inference

Run the baseline agent:
```bash
python inference.py
```

Run task baselines:
```bash
python baseline.py
```

## Requirements Compliance

✅ Real-world task (crisis response coordination)  
✅ Full OpenEnv spec (typed models, step/reset/state, openenv.yaml)  
✅ 3 tasks with agent graders (easy→medium→hard, 0.0-1.0 scores)  
✅ Meaningful reward with partial progress signals  
✅ Baseline inference with reproducible scores  
✅ HF Spaces + working Dockerfile  
✅ Complete README with description, spaces, setup

## Project Structure

```
├── baseline.py           # Run task baselines (easy/medium/hard)
├── inference.py          # Advanced inference with logging
├── client.py             # OpenEnv client for connecting to server
├── server/
│   ├── app.py           # FastAPI server (main entry point)
│   ├── models.py        # Pydantic models (MyAction, MyObservation)
│   ├── my_env_environment.py  # Environment logic
│   └── Dockerfile       # Docker container definition
├── tasks/
│   ├── task_easy.py     # Easy: Classify incident urgency
│   ├── task_medium.py   # Medium: Allocate resources
│   └── task_hard.py     # Hard: Crisis coordination
├── requirements.txt     # Python dependencies
├── openenv.yaml         # OpenEnv configuration
└── Readme.md           # This file
```

## Project Files

- **baseline.py**: Tests LLM performance on all three tasks (easy, medium, hard) and reports average score
- **inference.py**: Advanced inference with detailed logging and step-by-step tracking
- **client.py**: OpenEnv HTTP client for connecting to the server
- **server/app.py**: FastAPI server implementing crisis response environment
- **server/models.py**: Pydantic data models for actions and observations
- **server/my_env_environment.py**: Core environment simulation logic
- **tasks/**: Task definitions with grading functions

## Running Examples

### Quick Test with Baseline

Test the LLM against all three tasks:
```bash
python baseline.py
```

Output:
```
=== easy Task ===
Model Output: medium
Score: 1.0

=== FINAL SCORE ===
0.75
```

### Advanced Inference

Run inference with detailed logging:
```bash
python inference.py
```

### Server Only

Run just the server for custom client integration:
```bash
python server/app.py
```
Server will be available at `http://localhost:7860`
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/my_env_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Baseline Inference Script

The repository includes a ready-to-run inference script that uses the OpenAI client and prints the required structured logs.

```bash
python inference.py
```

Required environment variables:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN` or `API_KEY`

The script will emit:
- `[START] ...`
- `[STEP] ...`
- `[END] ...`

## Project Structure

```
my_env/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # MyEnv client
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── my_env_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```
