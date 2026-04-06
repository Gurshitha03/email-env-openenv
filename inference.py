import os
from openai import OpenAI
from http.server import BaseHTTPRequestHandler, HTTPServer
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


def run_tasks():

    for task in TASKS:

        env = EmailEnv(task=task)
        obs = env.reset()

        print(f"[START] task={task} env=email-env model={MODEL_NAME}")

        rewards = []
        step_count = 0

        try:
            while True:
                step_count += 1
                email = obs["email"]

                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{
                        "role": "user",
                        "content": f"""
You are an AI email assistant.

Classify the email into EXACTLY ONE WORD:
spam, important, urgent

Also internally decide action:
spam -> delete
important -> mark-read
urgent -> notify

Return ONLY ONE WORD.

Email: {email}
"""
                    }]
                )

                action = response.choices[0].message.content.strip().lower()
                action = action.split()[0]

                obs, reward, done, info = env.step({"category": action})
                rewards.append(reward)

                print(
                    f"[STEP] step={step_count} action={action} "
                    f"reward={reward:.2f} done={str(done).lower()} error=null"
                )

                if done:
                    break

            final_score = grade(task, env.predictions)
            success = final_score > 0.6

        except Exception as e:
            success = False
            print(
                f"[STEP] step={step_count} action=error "
                f"reward=0.00 done=true error={str(e)}"
            )

        rewards_str = ",".join([f"{r:.2f}" for r in rewards])

        print(
            f"[END] success={str(success).lower()} "
            f"steps={step_count} rewards={rewards_str}"
        )


# ----------------------------
# Simple UI Server
# ----------------------------
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"""
        <h2>Email Triage OpenEnv Running</h2>
        <p>Check the Logs tab for evaluation output.</p>
        """)


if __name__ == "__main__":

    # run once
    run_tasks()

    # keep space alive
    PORT = 7860
    print(f"Serving at port {PORT}")

    HTTPServer(("", PORT), Handler).serve_forever()