import asyncio
import os
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI

from tasks.task_easy import create_easy_task
from tasks.task_medium import create_medium_task
from tasks.task_hard import create_hard_task

# ---------------- ENV ----------------
load_dotenv()

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

if not API_KEY:
    raise ValueError("HF_TOKEN or API_KEY required")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ---------------- HARD CLAMP ----------------
def clamp_score(value: float) -> float:
    """
    Force score strictly into (0,1)
    """
    return round(min(0.9999, max(0.0001, float(value))), 4)

# ---------------- LOGGING ----------------
def log_start(task: str):
    print(f"[START] task={task} env=crisis_response_env model={MODEL_NAME}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    safe_reward = clamp_score(reward)
    error_val = error if error else "null"

    print(
        f"[STEP] step={step} action={action} reward={safe_reward:.4f} done={str(done).lower()} error={error_val}",
        flush=True
    )

def log_end(task: str, success: bool, steps: int, rewards: List[float]):
    safe_rewards = [clamp_score(r) for r in rewards]
    rewards_str = ",".join(f"{r:.4f}" for r in safe_rewards)

    final_score = clamp_score(sum(safe_rewards) / len(safe_rewards)) if safe_rewards else 0.0001

    print(
        f"[END] task={task} score={final_score:.4f} steps={steps} rewards={rewards_str}",
        flush=True
    )

# ---------------- LLM ----------------
def get_llm_output(observation: Dict):
    task_type = observation.get("task")

    if task_type == "classify_urgency":
        format_hint = "Return ONLY one word: low, medium, or high"
    elif task_type == "allocate_resources":
        format_hint = 'Return ONLY JSON list like ["ambulance","fire_truck"]'
    else:
        format_hint = 'Return ONLY JSON like {"plan":[{"incident_id":1,"resources":["fire_truck"]}]}'

    prompt = f"""
You are an expert crisis response system.

{format_hint}

Situation:
{observation}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200
        )

        output = response.choices[0].message.content.strip()

        try:
            return json.loads(output), None
        except:
            return output, f"invalid_output={output}"

    except Exception as e:
        return None, str(e)

# ---------------- RUN TASK ----------------
def run_task(name, task):
    log_start(name)

    rewards = []

    try:
        observation = task.get_observation()

        output, error = get_llm_output(observation)

        if output is None:
            # HARD FAIL PATH
            log_step(1, "error", 0.0001, True, error)
            log_end(name, False, 1, [0.0001])
            return 0.0001

        raw_score = task.grade(output)
        score = clamp_score(raw_score)

        action_str = str(output).replace("\n", "")

        log_step(
            step=1,
            action=action_str,
            reward=score,
            done=True,
            error=error
        )

        rewards.append(score)

        log_end(
            task=name,
            success=score > 0.1,
            steps=1,
            rewards=rewards
        )

        return score

    except Exception as e:
        # HARD CRASH SAFETY
        print(f"[STEP] step=1 action=error reward=0.0001 done=true error={str(e)}", flush=True)
        print(f"[END] task={name} score=0.0001 steps=1 rewards=0.0001", flush=True)
        return 0.0001

# ---------------- MAIN ----------------
async def main():
    tasks = [
        ("easy", create_easy_task()),
        ("medium", create_medium_task()),
        ("hard", create_hard_task())
    ]

    total = 0.0

    for name, task in tasks:
        score = run_task(name, task)
        total += score

    final_score = clamp_score(total / len(tasks))
    print(f"\nFINAL SCORE: {final_score:.4f}")

if __name__ == "__main__":
    asyncio.run(main())