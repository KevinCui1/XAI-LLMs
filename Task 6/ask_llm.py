import json
import os
from mlx_lm import load, generate

MODEL = "mlx-community/Qwen3-4B-4bit"
# Write results next to this script (the Task 6 folder), regardless of the
# directory you launch it from.
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")

model, tokenizer = load(MODEL)       
results = []

def ask(user_msg, system=None, max_tokens=512, think=True):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_msg})
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=think)
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)

# Qwen has a thinking mode, so if the model invokes it, then this will retrieve all of the reasoning
def split_reasoning(text):
    if "</think>" in text:
        reasoning, answer = text.split("</think>", 1)
        return reasoning.replace("<think>", "").strip(), answer.strip()
    return "", text.strip()


def record(category, question, text, answer=None):
    reasoning, final = split_reasoning(text)
    entry = {
        "category": category,
        "question": question,
        "reasoning": reasoning,
        "answer": answer if answer is not None else final,
    }
    results.append(entry)

    
    print(f"[{category}]")
    print(f"Q: {question}")
    if reasoning:
        print(f"\n(reasoning)\n{reasoning}")
    print(f"\nA: {entry['answer']}")
    
open_ended = [
    "Summarize the key causes that led to the decline of the Harappan Civilization.",
    "Explain how a large language model is able to gain knowledge and respond to questions correctly, compared to just being able to forulate coherent sentences.",
    "Give me a brief summary of the plot of Jujutsu Kaisen, no spoilers please!",
]

print("3 Open-Ended Questions\n")
for q in open_ended:
    text = ask(q, max_tokens=600, think=True)
    record("open-ended", q, text)


prompt_engineering = [
    ("""PLease identify and label the sentiment of different product review as one of either Positive, Negative, or Neutral.
Examples:
Review: "This blender broke after two days." -> Negative
Review: "Works exactly as described, very happy." -> Positive
Review: "It's fine, nothing special." -> Neutral""",
     'Review: "The battery drains way too fast, but the screen is gorgeous." -> ?'),

    ("""Define the operator # as: a # b = (a + b) * 2.
Examples:
3 # 4 = 14
1 # 1 = 4
10 # 0 = 20""",
     "Using the same rule, what is 6 # 5?"),

    ("""Convert Celsius to Fahrenheit using: F = C * 1.8 + 32.
Examples:
0 C -> 32 F
100 C -> 212 F
25 C -> 77 F""",
     "Using the same method, convert 40 C to F."),
]

print("\n3 Prompt-Engineering Reasoning Questions\n")
for system, q in prompt_engineering:
    text = ask(q, system=system, max_tokens=512, think=True)
    record("prompt-engineering", q, text)


single_word = [
    "Is 91 a prime number? Only answer one word: True or False.",
    "Do humans share common ancestry with certain bird species? Only answer one word: Yes or No.",
    "Which planet is known as the Red Planet? Only answer one word.",
]

print("\n3 Single-Word Reasoning Questions\n")
for q in single_word:
    text = ask(q, max_tokens=1024, think=True)
    reasoning, answer = split_reasoning(text)
    tokens = answer.split()
    word = tokens[0].strip('.,!?"\'') if tokens else ""
    record("single-word", q, text, answer=word)


with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)
