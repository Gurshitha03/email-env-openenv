---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
tags:
- openenv
pinned: false
---

# 📧 Email Triage OpenEnv

An interactive **OpenEnv-based email triage environment** that simulates real-world email classification and decision-making.

The environment presents incoming emails to an AI agent, which must classify each email as:

- `spam`
- `important`
- `urgent`

The project includes an interactive web interface for testing the environment directly in the Hugging Face Space.

---

## 🚀 Overview

This project simulates a real-world **email triage system** used in domains such as:

- Customer support automation
- Banking and financial alerts
- Spam detection systems
- Cybersecurity monitoring
- Automated inbox management

An AI agent processes incoming emails, classifies them, and receives rewards based on the quality of its decisions.

---

## 🖥️ Interactive Web UI

The project includes a browser-based interface for interacting with the OpenEnv environment.

### UI Features

- 📧 Incoming email display
- 🎯 Difficulty selection
- 🗑️ Spam classification
- ⭐ Important classification
- 🚨 Urgent classification
- 🏆 Real-time reward display
- 📊 Classification progress
- 🔄 Multi-step email interaction
- 🎉 Task completion feedback

The interface is directly connected to the OpenEnv environment through the backend API.

---

## 🎯 Task Design

The environment is divided into three difficulty levels:

### 🟢 Easy — Spam Detection

Detect promotional, suspicious, and scam emails.

### 🟡 Medium — Important Email Detection

Identify useful or important communication such as:

- Meetings
- Order updates
- Project deadlines
- Bills
- Account-related communication

### 🔴 Hard — Urgent / Security Detection

Identify urgent security-related emails such as:

- OTP alerts
- Suspicious login activity
- Unauthorized transactions
- Security warnings

Each task contains multiple steps simulating a real inbox stream.

---

## ⚙️ Observation Space

Each observation returned by the environment contains an email:

```json
{
  "email": "Your OTP for ₹10,000 transaction is 839201"
}
```

---

## 🧠 Decision Mapping

Each classification corresponds to a real-world action:

- **spam** → Delete
- **important** → Mark as read
- **urgent** → Notify user

This allows the environment to model how an AI agent's classification can lead to different downstream actions.

---

## 🏆 Reward Design

The reward function provides trajectory-level feedback.

- **Correct classification** → High reward
- **Important predicted as urgent** → Partial reward
- **Urgent predicted as important** → Partial reward
- **Other incorrect predictions** → Low reward

This provides meaningful feedback instead of treating every prediction equally.

---

## 🔄 Environment Features

- Multi-step interaction
- Randomized email ordering
- Realistic email examples
- Deterministic grading
- Partial reward shaping
- Difficulty-based tasks
- Generalization-friendly prompts
- Interactive browser interface
- REST-style environment endpoints
- Docker-based deployment

---

## 🏗️ Architecture

```text
                 ┌──────────────────────┐
                 │   Interactive UI     │
                 │    index.html        │
                 └──────────┬───────────┘
                            │
                            │ HTTP
                            ▼
                 ┌──────────────────────┐
                 │     API Server       │
                 │     server/app.py    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │      EmailEnv        │
                 │      my_env.py       │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Observation + Reward │
                 └──────────────────────┘
```

---

## 🔄 Interaction Flow

```text
Start Task
    ↓
Select Difficulty
    ↓
Receive Email Observation
    ↓
Classify Email
    ↓
Environment Calculates Reward
    ↓
Receive Next Email
    ↓
Repeat
    ↓
Task Completed
```

---

## 📊 Tasks

### Easy — Spam Detection

Detect promotional and scam emails.

**Example:**

```text
Congratulations! You won a $1000 gift card. Click now!
```

**Expected classification:** `spam`

---

### Medium — Important Email Detection

Identify important communication such as meetings and updates.

**Example:**

```text
Your electricity bill is due tomorrow. Please pay on time.
```

**Expected classification:** `important`

---

### Hard — Urgent / Security Detection

Detect urgent security alerts and suspicious activity.

**Example:**

```text
Security alert: New login detected from unknown device.
```

**Expected classification:** `urgent`

---

## 🧪 Example Interaction

```text
Incoming Email:
"Security alert: New login detected from unknown device."

Agent:
urgent

Reward:
0.894

Progress:
1 / 5
```

The environment then provides the next email until the task is completed.

---

## 🐳 Docker

The environment is packaged using Docker for reproducible deployment.

**Technologies:**

- Python 3.10
- Docker
- Port 7860

---

## ▶️ Running Locally

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the baseline inference:

```bash
python inference.py
```

The interactive server can be started through the Docker/OpenEnv configuration.

---

## 📈 Baseline Scores

Baseline inference using:

```text
Qwen/Qwen2.5-72B-Instruct
```

Results:

```text
Easy: 1.00
Medium: ~0.76
Hard: 1.00

Average Score: ~0.92
```

Scores may vary slightly due to randomized email ordering.

---

## 📁 Project Structure

```text
email-env/
│
├── server/
│   ├── app.py
│   └── index.html
│
├── inference.py
├── my_env.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── README.md
└── uv.lock
```

---

## 🌟 Project Highlights

- 🤗 Built as an OpenEnv environment
- 📧 Real-world email triage simulation
- 🤖 Designed for AI-agent interaction
- 🧠 Multi-level classification tasks
- 🏆 Reward-based evaluation
- 🖥️ Interactive web UI
- 🐳 Dockerized deployment
- 🔄 Multi-step trajectories
- 📊 Baseline evaluation with Qwen

---

## 📌 Project Status

**Status: Completed and Interactive**

The environment, backend API, reward system, and browser-based interface are integrated and deployed as a Hugging Face Space.
---

## 🤗 Live Demo

The interactive version of this project is deployed on Hugging Face Spaces.

👉 [Try Email Triage OpenEnv on Hugging Face](https://huggingface.co/spaces/Gurshitha/email-env)
