import os
import json
from dotenv import load_dotenv
from openai import OpenAI

from tasks.task_easy import create_easy_task
from tasks.task_medium import create_medium_task
from tasks.task_hard import create_hard_task

# Load env variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY or HF_TOKEN or API_KEY is required in .env")

client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)

def run_task(task):
    observation = task.get_observation()

    prompt = f"""
You are an expert crisis response coordinator.

Given the situation below, make the best decision.

IMPORTANT:
- Return ONLY the answer
- No explanations
- Follow expected format strictly

Situation:
{observation}
"""
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0
    )

    output = (response.choices[0].message.content or "").strip()

    try:
        parsed_output = json.loads(output)
    except json.JSONDecodeError:
        parsed_output = output

    score = task.grade(parsed_output)
    return output, score

def  main():
    tasks = [
        ("easy",create_easy_task()),
        ("medium", create_medium_task()),
        ("hard", create_hard_task())
    ]

    total_score = 0

    for name, task in tasks:
        print(f"\n=== {name} Task ===")
        
        output, score = run_task(task)
        
        print("Model Output:", output)
        print("Score:", score)
        
        total_score += score
        
    final_score = total_score / len(tasks)

    print("\n=== FINAL SCORE ===")
    print(final_score)


if __name__ == "__main__":
    main()
    

    
    
