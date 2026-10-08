"""
The smallest possible MLOps habit: a fixed set of questions, run through
the exact same pipeline used in production (app.rag.answer_question), with
a crude keyword-overlap score.

This is deliberately simple. As you grow this, the natural next steps are:
- use an LLM to grade answer quality instead of keyword matching
- track scores over time (one row per run: date, config, score)
- fail a CI step if the score drops below a threshold after a change

Run: python eval/run_eval.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import rag  # noqa: E402


def score_answer(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return float("nan")  # no ground truth to check against yet
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)


def main():
    eval_path = os.path.join(os.path.dirname(__file__), "eval_set.json")
    with open(eval_path) as f:
        cases = json.load(f)

    print(f"Running {len(cases)} eval cases...\n")
    scores = []

    for i, case in enumerate(cases, 1):
        result = rag.answer_question(case["question"])
        score = score_answer(result["answer"], case.get("expected_keywords", []))
        if score == score:  # not NaN
            scores.append(score)

        print(f"[{i}] Q: {case['question']}")
        print(f"    A: {result['answer'][:200]}{'...' if len(result['answer']) > 200 else ''}")
        print(f"    Sources: {result['sources']}")
        print(f"    Latency: {result['latency_ms']} ms")
        print(f"    Keyword score: {score if score == score else 'N/A (no keywords set)'}\n")

    if scores:
        print(f"Average keyword score: {sum(scores) / len(scores):.2f} over {len(scores)} scored cases")
    else:
        print("No scored cases yet — fill in expected_keywords in eval_set.json.")


if __name__ == "__main__":
    main()
