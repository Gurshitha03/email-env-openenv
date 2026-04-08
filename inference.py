import os
from openai import OpenAI
from my_env import EmailEnv, grade

# ----------------------------
# Setup Client
# ----------------------------
client = OpenAI(
    base_url=os.getenv("API_BASE_URL"),
    api_key=os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
)

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASKS = ["easy", "medium", "hard"]


def clamp(x):
    eps = 1e-6
    return max(eps, min(x, 1 - eps))


# ----------------------------
# SAFE LLM CALL
# ----------------------------
def classify_email(email):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            timeout=10,
            max_tokens=2,
            temperature=0,
            messages=[{
                "role": "user",
                "content": f"Classify into one word: spam, important, urgent\nEmail: {email}"
            }]
        )

        action = response.choices[0].message.content.strip().lower()
        return action.split()[0]

    except Exception:
        return "important"


# ----------------------------
# Run tasks
# ----------------------------
for task in TASKS:

    env = EmailEnv(task=task)
    obs = env.reset()

    print(f"[START] task={task} env=email-env model={MODEL_NAME}")

    rewards = []
    step_count = 0

    try:
        while True:
            step_count += 1

            action = classify_email(obs["email"])
            obs, reward, done, info = env.step({"category": action})

            reward = clamp(reward)
            rewards.append(reward)

            print(
                f"[STEP] step={step_count} action={action} "
                f"reward={reward:.2f} done={str(done).lower()} error=null"
            )

            if done:
                break

        final_score = clamp(grade(task, env.predictions))
        success = final_score > 0.6

    except Exception as e:
        success = False
        safe_reward = 0.000001

        print(
            f"[STEP] step={step_count} action=error "
            f"reward={safe_reward:.2f} done=true error={str(e)}"
        )

    rewards_str = ",".join([f"{r:.2f}" for r in rewards])

    print(
        f"[END] success={str(success).lower()} "
        f"steps={step_count} rewards={rewards_str}"
    )