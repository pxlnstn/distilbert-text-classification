"""Train several configs (epochs x learning rate) and compare them.

Writes results/results.csv plus the charts, and copies the best model to
models/distilbert-ticket-best. Each config runs in its own subprocess so the GPU/RAM
is freed between runs. Pass --full for the whole dataset.
"""

from __future__ import annotations

import argparse

import bigstack

# cfg1-4 vary epochs and lr; cfg5 adds class weighting (compare with cfg3);
# cfg6-9 just train longer at the best lr
CONFIGS = [
    {"name": "cfg1_ep1_lr5e-5", "epochs": 1, "lr": 5e-5},
    {"name": "cfg2_ep2_lr5e-5", "epochs": 2, "lr": 5e-5},
    {"name": "cfg3_ep3_lr3e-5", "epochs": 3, "lr": 3e-5},
    {"name": "cfg4_ep3_lr2e-5", "epochs": 3, "lr": 2e-5},
    {"name": "cfg5_ep3_lr3e-5_weighted", "epochs": 3, "lr": 3e-5, "class_weights": True},
    {"name": "cfg6_ep4_lr3e-5", "epochs": 4, "lr": 3e-5},
    {"name": "cfg7_ep5_lr3e-5", "epochs": 5, "lr": 3e-5},
    {"name": "cfg8_ep6_lr3e-5", "epochs": 6, "lr": 3e-5},
    {"name": "cfg9_ep7_lr3e-5", "epochs": 7, "lr": 3e-5},
]


def main() -> None:
    import json
    import os
    import shutil
    import subprocess
    import sys
    import time

    import matplotlib
    matplotlib.use("Agg")  # just write PNGs, no window
    import matplotlib.pyplot as plt
    import pandas as pd

    p = argparse.ArgumentParser(description="Run the DistilBERT config comparison.")
    p.add_argument("--subset", type=int, default=8000,
                   help="Training examples per run (default 8000). Ignored with --full.")
    p.add_argument("--full", action="store_true",
                   help="Use the entire training split (slower, best final numbers).")
    args = p.parse_args()

    os.makedirs("results", exist_ok=True)
    exp_root = os.path.join("models", "experiments")
    os.makedirs(exp_root, exist_ok=True)

    child_env = dict(os.environ)

    MAX_ATTEMPTS = 3  # this GPU occasionally throws a transient "CUDA unknown error"

    # tag dirs by data size so --full and subset runs don't overwrite each other
    data_tag = "full" if args.full else f"n{args.subset}"

    rows = []
    for cfg in CONFIGS:
        out_dir = os.path.join(exp_root, f"{cfg['name']}__{data_tag}")
        metrics_path = os.path.join(out_dir, "metrics.json")

        # Resume: if this config already finished, don't retrain it.
        if os.path.exists(metrics_path):
            print(f"\n[skip] {cfg['name']} already trained (found {metrics_path}).")
        else:
            print("\n" + "#" * 70)
            print(f"# TRAINING {cfg['name']}  (epochs={cfg['epochs']}, lr={cfg['lr']}"
                  f"{', weighted' if cfg.get('class_weights') else ''})")
            print("#" * 70, flush=True)

            cmd = [
                sys.executable, os.path.join("src", "train.py"),
                "--epochs", str(cfg["epochs"]),
                "--lr", str(cfg["lr"]),
                "--out", out_dir,
                "--test",  # also evaluate on the held-out test split
            ]
            if cfg.get("class_weights"):
                cmd += ["--class_weights"]
            if not args.full:
                cmd += ["--subset", str(args.subset)]

            for attempt in range(1, MAX_ATTEMPTS + 1):
                result = subprocess.run(cmd, env=child_env)
                if result.returncode == 0 and os.path.exists(metrics_path):
                    break
                if attempt < MAX_ATTEMPTS:
                    print(f"\n[retry] {cfg['name']} failed (attempt {attempt}/"
                          f"{MAX_ATTEMPTS}, likely a transient CUDA error). Retrying...",
                          flush=True)
                    time.sleep(5)
            else:
                raise SystemExit(
                    f"{cfg['name']} failed {MAX_ATTEMPTS} times. The GPU keeps throwing "
                    "errors - close other GPU/RAM-heavy apps (browser, WSL, Docker) and "
                    "re-run; finished configs are skipped automatically."
                )

        with open(metrics_path, encoding="utf-8") as f:
            m = json.load(f)
        rows.append({
            "config": cfg["name"],
            "epochs": cfg["epochs"],
            "lr": cfg["lr"],
            "max_length": m.get("max_length", ""),
            "class_weights": m.get("class_weights", False),
            "train_size": m["train_size"],
            "eval_loss": round(m["eval_loss"], 4),
            "eval_accuracy": round(m["eval_accuracy"], 4),
            "eval_top2_accuracy": round(m.get("eval_top2_accuracy", float("nan")), 4),
            "eval_f1_macro": round(m["eval_f1_macro"], 4),
            "test_accuracy": round(m.get("test_accuracy", float("nan")), 4),
            "test_top2_accuracy": round(m.get("test_top2_accuracy", float("nan")), 4),
            "test_f1_macro": round(m.get("test_f1_macro", float("nan")), 4),
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join("results", "results.csv")
    df.to_csv(csv_path, index=False)

    # ---- charts -------------------------------------------------------------
    labels = df["config"].tolist()

    plt.figure(figsize=(9, 5))
    x = range(len(labels))
    plt.bar([i - 0.2 for i in x], df["eval_accuracy"], width=0.4, label="validation")
    plt.bar([i + 0.2 for i in x], df["test_accuracy"], width=0.4, label="test")
    plt.xticks(list(x), labels, rotation=20, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per configuration")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("results", "accuracy.png"), dpi=130)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(labels, df["eval_loss"], color="tab:red")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Validation loss")
    plt.title("Validation loss per configuration (lower is better)")
    plt.tight_layout()
    plt.savefig(os.path.join("results", "loss.png"), dpi=130)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar([i - 0.2 for i in x], df["eval_f1_macro"], width=0.4, label="validation")
    plt.bar([i + 0.2 for i in x], df["test_f1_macro"], width=0.4, label="test")
    plt.xticks(list(x), labels, rotation=20, ha="right")
    plt.ylabel("Macro-F1")
    plt.title("Macro-F1 per configuration (treats every class equally)")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("results", "f1_macro.png"), dpi=130)
    plt.close()

    # ---- pick + copy the best model ----------------------------------------
    best_idx = int(df["eval_accuracy"].idxmax())
    best = df.iloc[best_idx]
    best_name = best["config"]
    best_dir = os.path.join("models", "distilbert-ticket-best")
    if os.path.exists(best_dir):
        shutil.rmtree(best_dir)
    shutil.copytree(os.path.join(exp_root, f"{best_name}__{data_tag}"), best_dir)

    # ---- report -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS  (also saved to results/results.csv)")
    print("=" * 70)
    print(df.to_string(index=False))
    print("\nCharts written: results/accuracy.png , results/loss.png , "
          "results/f1_macro.png")
    print("=" * 70)
    print("BEST CONFIGURATION (by validation accuracy):")
    print(f"  {best_name}: epochs={best['epochs']}, lr={best['lr']}")
    print(f"  validation accuracy={best['eval_accuracy']}, "
          f"macro-F1={best['eval_f1_macro']}, loss={best['eval_loss']}")
    print(f"  test accuracy={best['test_accuracy']}, macro-F1={best['test_f1_macro']}")
    print(f"  -> copied to {best_dir} (used by the agent in Part E)")
    print("=" * 70)


if __name__ == "__main__":
    bigstack.run(main)
