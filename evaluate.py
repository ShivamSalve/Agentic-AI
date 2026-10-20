"""
evaluate.py  --  Measures how accurate the system is on a small labelled test set.
Run:  python evaluate.py      (creates results.csv - use the numbers in your report!)
"""
import csv
from time import perf_counter

from agents import MOCK, AgentError
from workflow import run

# Canned mock responses cannot produce a meaningful accuracy score.
if MOCK:
    raise SystemExit("Evaluation requires real model responses. Set MOCK_MODE=0 and configure LLM_API_KEY.")

# (claim, is_it_true?)
TEST_SET = [
    ("The Earth orbits the Sun.", True),
    ("Water boils at 100 degrees Celsius at sea level.", True),
    ("New Delhi is the capital of India.", True),
    ("Mount Everest is the highest mountain above sea level.", True),
    ("The Great Wall of China is visible from the Moon with the naked eye.", False),
    ("Humans only use 10 percent of their brains.", False),
    ("Vaccines cause autism.", False),
    ("Drinking bleach cures COVID-19.", False),
    ("Lightning never strikes the same place twice.", False),
    ("The Pacific Ocean is the largest ocean on Earth.", True),
]

rows, correct = [], 0
for claim, truth in TEST_SET:
    start = perf_counter()
    state = run(claim)
    secs = round(perf_counter() - start, 1)
    predicted_true = state["overall"] == "True"
    ok = state["overall"] != "Unverified" and predicted_true == truth
    correct += ok
    rows.append([claim, "True" if truth else "False", state["overall"], "yes" if ok else "NO", secs])
    print(f"{'OK ' if ok else 'ERR'} | expected={truth!s:5} | got={state['overall']:10} | {secs}s | {claim}")

if not rows:
    raise AgentError("The evaluation test set is empty.")
acc = correct / len(TEST_SET) * 100
avg = sum(r[-1] for r in rows) / len(rows)
print(f"\nAccuracy: {correct}/{len(TEST_SET)} = {acc:.0f}%   |   Avg time per claim: {avg:.1f}s")

with open("results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["claim", "expected", "predicted", "correct", "seconds"])
    w.writerows(rows)
print("Saved results.csv")
