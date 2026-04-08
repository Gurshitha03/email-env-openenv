from pydantic import BaseModel
import random

# ----------------------------
# Models
# ----------------------------
class Observation(BaseModel):
    email: str


class Action(BaseModel):
    category: str
    response: str


class Reward(BaseModel):
    score: float


# ----------------------------
# Environment
# ----------------------------
class EmailEnv:
    def __init__(self, task="easy"):
        self.task = task

        if task == "easy":
            self.emails = [
                {"text": "Congratulations! You won a $1000 gift card. Click now!", "label": "spam"},
                {"text": "Exclusive deal! Buy now and get 80% discount!", "label": "spam"},
                {"text": "You have been selected for a lottery prize. Claim immediately!", "label": "spam"},
                {"text": "Earn money from home with no investment. Join today!", "label": "spam"},
                {"text": "Limited-time cashback offer! Click to claim your reward.", "label": "spam"}
            ]

        elif task == "medium":
            self.emails = [
                {"text": "Reminder: Team meeting scheduled at 5 PM today.", "label": "important"},
                {"text": "Your Amazon order has been shipped and will arrive tomorrow.", "label": "important"},
                {"text": "Project deadline is tomorrow. Please submit your report.", "label": "important"},
                {"text": "Your electricity bill is due tomorrow. Please pay on time.", "label": "important"},
                {"text": "Please verify your email address to continue using services.", "label": "important"}
            ]

        elif task == "hard":
            self.emails = [
                {"text": "Your OTP for ₹10,000 transaction is 839201. Do not share it.", "label": "urgent"},
                {"text": "Alert: ₹5000 debited from your account. If not you, contact support immediately.", "label": "urgent"},
                {"text": "Security alert: New login detected from unknown device.", "label": "urgent"},
                {"text": "Your account password was changed. If not you, reset immediately.", "label": "urgent"},
                {"text": "Multiple failed login attempts detected. Account temporarily locked.", "label": "urgent"}
            ]

        else:
            raise ValueError("Invalid task")

        random.shuffle(self.emails)
        self.index = 0
        self.predictions = []

    def reset(self):
        self.index = 0
        self.predictions = []
        return {"email": self.emails[self.index]["text"]}

    def state(self):
        return {
            "task": self.task,
            "index": self.index,
            "predictions": self.predictions
        }

    def step(self, action):
        correct = self.emails[self.index]["label"]
        pred = action["category"]

        self.predictions.append(pred)

        # reward logic
        if pred == correct:
            reward = 0.9
        elif correct == "urgent" and pred == "important":
            reward = 0.7
        elif correct == "important" and pred == "urgent":
            reward = 0.6
        else:
            reward = 0.1

        # add jitter (avoid constant scores)
        jitter = random.uniform(-0.02, 0.02)
        reward = reward + jitter

        # clamp strictly between (0,1)
        eps = 1e-6
        reward = max(eps, min(reward, 1 - eps))

        self.index += 1
        done = self.index >= len(self.emails)

        obs = {"email": self.emails[self.index]["text"]} if not done else {}

        return obs, reward, done, {"correct": correct}


# ----------------------------
# Grader
# ----------------------------
def grade(task, predictions):
    correct_answers = {
        "easy": ["spam"] * 5,
        "medium": ["important"] * 5,
        "hard": ["urgent"] * 5
    }

    correct = correct_answers[task]
    score = 0.0

    for p, c in zip(predictions, correct):
        if p == c:
            reward = 0.9
        elif c == "urgent" and p == "important":
            reward = 0.7
        elif c == "important" and p == "urgent":
            reward = 0.6
        else:
            reward = 0.1

        # add jitter
        reward += random.uniform(-0.02, 0.02)

        # clamp
        eps = 1e-6
        reward = max(eps, min(reward, 1 - eps))

        score += reward

    final_score = score / len(correct)

    # final clamp
    eps = 1e-6
    final_score = max(eps, min(final_score, 1 - eps))

    return float(f"{final_score:.6f}")