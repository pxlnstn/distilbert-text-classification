"""Load the fine-tuned classifier and expose classify_ticket().

    classify_ticket("my internet is down") -> {"label": ..., "confidence": ..., "scores": ...}

The CrewAI agent imports classify_ticket() and uses it as a tool. Can also be run
from the command line:  python src/classifier_tool.py "double charged on my invoice"
"""

from __future__ import annotations

import os

import bigstack

MODEL_DIR = os.path.join("models", "distilbert-ticket-best")

_pipe = None  # cached pipeline so we only load the model once


def _get_pipe():
    global _pipe
    if _pipe is None:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline,
        )

        if not os.path.isdir(MODEL_DIR):
            raise FileNotFoundError(
                f"Trained model not found at '{MODEL_DIR}'. "
                "Run 'python src/run_experiments.py' first to create it."
            )
        # eager attention - same fix as training
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR, attn_implementation="eager"
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        device = 0 if torch.cuda.is_available() else -1
        _pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            top_k=None,  # return scores for ALL classes
        )
    return _pipe


def classify_ticket(text: str) -> dict:
    """Predict the department queue for one ticket. Returns the top label, its
    confidence, and the full score distribution."""
    pipe = _get_pipe()
    # wrap in a list so we get a list-of-lists back
    all_scores = pipe([text], truncation=True, max_length=256)[0]
    all_scores = sorted(all_scores, key=lambda d: d["score"], reverse=True)
    top = all_scores[0]
    return {
        "label": top["label"],
        "confidence": float(top["score"]),
        "scores": {d["label"]: float(d["score"]) for d in all_scores},
    }


def main() -> None:
    import sys

    text = " ".join(sys.argv[1:]).strip()
    if not text:
        text = "My internet connection keeps dropping every few minutes since this morning."
        print(f"(no ticket given on command line, using example)\n")

    result = classify_ticket(text)
    print("Ticket:", text)
    print("-" * 60)
    print(f"Predicted department : {result['label']}")
    print(f"Confidence           : {result['confidence']:.2%}")
    print("Top 3 departments:")
    for label, score in list(result["scores"].items())[:3]:
        print(f"   {label:32s} {score:.2%}")


if __name__ == "__main__":
    bigstack.run(main)
