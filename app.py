import requests

MODEL = "qwen3.5:4b"
OLLAMA_URL = "http://localhost:11434/api/chat"


def ask(prompt, system=None, temperature=0.7, think=False):
    messages = []

    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "think": think,
            "options": {"temperature": temperature},
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


# ---------------------------------------------------------------------------
# 1) Open-ended questions (long, open answers)
# ---------------------------------------------------------------------------
OPEN_ENDED = [
    "Summarize the events of the Triangle Shirtwaist Factory fire",
    "Describe what the AI bubble is 2026",
    "Describe the impacts of Covid-19 on society",
]

# ---------------------------------------------------------------------------
# 2) Reasoning with prompt engineering: give rules + examples, then ask
# ---------------------------------------------------------------------------
FEW_SHOT = [
    # Task A: made-up operator
"""
Rules: The operator # is defined as a # b = (a * 2) + b.
Examples:
3 # 4 = 10
5 # 1 = 11
0 # 7 = 7
Question: What is 6 # 3? Show your steps, then give the final answer.
""",
# (6 # 3 should be = 15)

    # Task B: sentiment classification
"""
Rules: Classify each review as POSITIVE, NEGATIVE, or MIXED.
MIXED means the review contains both a clear positive and a clear negative point.
Examples:
"The food was amazing and the waiters were friendly." -> POSITIVE
"The food was cold and the waiters were rude." -> NEGATIVE
"The food was great, but the service was slow." -> MIXED
Question: "The hotel room was spotless, though the wifi barely worked." ->
Explain your reasoning briefly, then give the label.
""",

    # Task C: pattern / sequence rule
"""
Rules: Each next number in a sequence is the previous number times 2, minus 1 (next = prev * 2 - 1).
Examples:
Start 3 -> 3, 5, 9, 17
Start 2 -> 2, 3, 5, 9
Question: Start 4 -> give the first 5 numbers of the sequence and explain how you got them.""",
]

# ---------------------------------------------------------------------------
# 3) Reasoning questions with a single-word answer
# ---------------------------------------------------------------------------
SINGLE_WORD_SYSTEM = (
    "Answer with exactly one word. Do not explain. "
    "For true/false questions answer only True or False."
)
SINGLE_WORD = [
    "True or False: The Earth is flat..",
    "True or False: If it rains, then the ground is wet. So if the ground is wet, it must have rained.",
    "Tom is taller than Ana. Ana is taller than Leo. Who is the shortest?",
]


def main():
    sections = [
        ("Open-ended questions", OPEN_ENDED, None, 0.7, False),
        ("Reasoning with rules and examples (few-shot)", FEW_SHOT, None, 0.2, True),
        ("Single-word reasoning answers", SINGLE_WORD, SINGLE_WORD_SYSTEM, 0.0, False),
    ]

    lines = [f"# Results from {MODEL}\n"]
    n = 1
    for title, questions, system, temp, think in sections:
        print(f"\n===== {title} =====")
        lines.append(f"\n## {title}\n")
        for q in questions:
            answer = ask(q, system=system, temperature=temp, think=think)
            print(f"\nQ{n}: {q}\n\nA{n}: {answer}\n" + "-" * 60)
            lines.append(f"### Q{n}\n```\n{q}\n```\n**Answer:**\n\n{answer}\n")
            n += 1

    with open("results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\nSaved all answers to results.md")


if __name__ == "__main__":
    main()