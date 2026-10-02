"""Day 5, Part D: compare two models on the same prompts.
Contains both Local Ollama (commented out) and Groq API (active)."""
import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

PROMPTS = [
    "Reply with exactly: OK",
    "In two sentences, what is an AI agent?",
    "A course costs Rs. 18,000 with a 15% scholarship. What is payable? Show the steps.",
]

# ==========================================
# 1. LOCAL OLLAMA CONFIGURATION (Commented)
# ==========================================
# BASE_OLLAMA = "http://localhost:11434"
# MODELS_OLLAMA = ["qwen2.5:1.5b", "qwen2.5:7b"]      # put YOUR two models here
#
# def run_ollama(model, prompt):
#     start = time.time()
#     data = requests.post(f"{BASE_OLLAMA}/api/generate",
#                          json={"model": model, "prompt": prompt, "stream": False},
#                          timeout=600).json()
#     elapsed = time.time() - start
#     tokens = data.get("eval_count", 0)
#     load_ms = data.get("load_duration", 0) / 1e6
#     return elapsed, tokens, tokens / elapsed if elapsed else 0, load_ms, data.get("response", "").strip()
#
# if __name__ == "__main__":
#     for model in MODELS_OLLAMA:
#         print("=" * 72)
#         print("MODEL:", model)
#         for prompt in PROMPTS:
#             elapsed, tokens, rate, load_ms, text = run_ollama(model, prompt)
#             print(f"\n  prompt: {prompt[:50]}")
#             print(f"  {elapsed:5.1f} s | {tokens:4d} tokens | {rate:5.1f} tok/s | load {load_ms:7.1f} ms")
#             print(f"  answer: {text[:160]}")
#         print()


# ==========================================
# 2. GROQ API CONFIGURATION (Active)
# ==========================================
BASE = "https://api.groq.com/openai/v1"
API_KEY = os.getenv("GROQ_API_KEY")
# Using the model from .env, and a standard fast model for comparison
MODELS = [
    os.getenv("MODEL", "openai/gpt-oss-120b"),
    os.getenv("MODEL2", "qwen/qwen3.8-27b"),
    os.getenv("MODEL3", "meta-llama/llama-prompt-guard-2-86m"),
    os.getenv("MODEL4", "canopylabs/orpheus-v1-english")
]

if not API_KEY:
    print("Error: GROQ_API_KEY not found in .env file.")
    exit(1)

def run_groq(model, prompt):
    start = time.time()
    
    response = requests.post(
        f"{BASE}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        },
        timeout=30
    ).json()
    
    elapsed = time.time() - start
    
    if "choices" in response:
        text = response["choices"][0]["message"]["content"].strip()
        usage = response.get("usage", {})
        tokens = usage.get("completion_tokens", 0)
    else:
        text = f"Error or unexpected response: {json.dumps(response)}"
        tokens = 0
        
    rate = tokens / elapsed if elapsed else 0
    load_ms = 0.0  # Groq models are always hot, no load time metric
    
    return elapsed, tokens, rate, load_ms, text

if __name__ == "__main__":
    for model in MODELS:
        print("=" * 72)
        print("MODEL:", model)
        for prompt in PROMPTS:
            elapsed, tokens, rate, load_ms, text = run_groq(model, prompt)
            print(f"\n  prompt: {prompt[:50]}")
            print(f"  {elapsed:5.1f} s | {tokens:4d} tokens | {rate:5.1f} tok/s | load {load_ms:7.1f} ms")
            print(f"  answer: {text[:160]}")
        print()
