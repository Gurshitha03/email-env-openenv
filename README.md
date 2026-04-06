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

# 📧 Email Triage OpenEnv Environment

## 🚀 Overview
This project simulates a real-world **email triage system** used in domains such as:
- Customer support automation  
- Banking and financial alerts  
- Spam detection systems  
- Cybersecurity monitoring  

An AI agent processes incoming emails, classifies them, and models decision-making behavior.

---

## 🎯 Task Design

The environment is divided into three difficulty levels:

- **Easy** → Spam detection  
- **Medium** → Important email detection  
- **Hard** → Urgent/security email detection  

Each task contains multiple steps simulating a real inbox stream.

---

## ⚙️ Observation Space

Each observation returned by the environment:

{
  "email": "Your OTP for ₹10,000 transaction is 839201"
}

---

## ⚙️ Action Space

The agent must output exactly one of:

- `spam`  
- `important`  
- `urgent`  

---

## 🧠 Decision Mapping

Each classification corresponds to a real-world action:

| Category | Action |
|----------|--------|
| spam | delete |
| important | mark-read |
| urgent | notify user |

---

## 🏆 Reward Design

The reward function provides trajectory-level signal:

- Correct classification → **1.0**
- Near miss (important ↔ urgent) → **0.6–0.7**
- Incorrect → **0.0**

This encourages meaningful partial progress.

---

## 🔄 Environment Features

- Multi-step interaction
- Randomized email ordering
- Realistic dataset
- Deterministic grading
- Partial reward shaping
- Generalization-friendly prompts

---

## 📊 Tasks

### Easy
Detect promotional and scam emails.

### Medium
Identify important communication such as meetings and updates.

### Hard
Detect urgent security alerts like OTP and suspicious activity.

---

## ▶️ Running Locally

```bash
python inference.py

## 📈 Baseline Scores

Baseline inference using `Qwen/Qwen2.5-72B-Instruct`

Easy: 1.00  
Medium: ~0.76  
Hard: 1.00  

Average Score: ~0.92

Scores may vary slightly due to randomized email ordering.