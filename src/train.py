"""Fine-tune distilbert-base-uncased to put a ticket in one of 10 queues.

Run it directly for a quick test, or import train_model() from run_experiments.py.
Heavy imports sit inside train_model() on purpose (see bigstack.py).
"""

from __future__ import annotations

import argparse

import bigstack

MODEL_NAME = "distilbert-base-uncased"


def train_model(
    epochs: float = 3,
    lr: float = 3e-5,
    subset: int | None = None,
    out_dir: str = "models/run",
    batch_size: int = 16,
    max_length: int = 256,
    seed: int = 42,
    save_model: bool = True,
    evaluate_test: bool = False,
    class_weights: bool = False,
) -> dict:
    """Fine-tune once, return metrics, optionally save the model.

    class_weights weights the loss by inverse class frequency (helps the rare queues).
    """
    import json
    import os

    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.utils.class_weight import compute_class_weight
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    from data_prep import ID2LABEL, LABEL2ID, NUM_LABELS, load_prepared_datasets

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available - aborting. Run 'python src/check_gpu.py' first."
        )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        # top-2: is the right label in the 2 best guesses?
        top2 = np.argsort(logits, axis=-1)[:, -2:]
        top2_acc = float(np.mean([labels[i] in top2[i] for i in range(len(labels))]))
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
            "top2_accuracy": top2_acc,
        }

    dd = load_prepared_datasets(subset=subset, seed=seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = dd.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # eager attention - the fused flash-attn kernel crashes on this GPU/driver
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        attn_implementation="eager",
    )

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        eval_strategy="epoch",
        save_strategy="no",        # save once at the end instead of per-epoch
        fp16=True,                 # mixed precision, fits 8 GB
        logging_steps=50,
        report_to="none",
        seed=seed,
        dataloader_num_workers=0,  # 0 workers is safer on Windows
        disable_tqdm=False,
    )

    # Optional class weighting (inverse frequency) to counter the imbalance.
    weight_tensor = None
    if class_weights:
        y = np.array(tokenized["train"]["label"])
        w = compute_class_weight("balanced", classes=np.arange(NUM_LABELS), y=y)
        weight_tensor = torch.tensor(w, dtype=torch.float)

    class WeightedTrainer(Trainer):
        """Trainer with a class-weighted cross-entropy loss."""

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = F.cross_entropy(
                outputs.logits, labels, weight=weight_tensor.to(outputs.logits.device)
            )
            return (loss, outputs) if return_outputs else loss

    trainer_cls = WeightedTrainer if class_weights else Trainer
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    val_metrics = trainer.evaluate(tokenized["validation"])

    result = {
        "epochs": epochs,
        "lr": lr,
        "max_length": max_length,
        "class_weights": class_weights,
        "train_size": tokenized["train"].num_rows,
        "eval_loss": float(val_metrics["eval_loss"]),
        "eval_accuracy": float(val_metrics["eval_accuracy"]),
        "eval_f1_macro": float(val_metrics["eval_f1_macro"]),
        "eval_top2_accuracy": float(val_metrics["eval_top2_accuracy"]),
    }

    if evaluate_test:
        test_metrics = trainer.evaluate(tokenized["test"])
        result["test_loss"] = float(test_metrics["eval_loss"])
        result["test_accuracy"] = float(test_metrics["eval_accuracy"])
        result["test_f1_macro"] = float(test_metrics["eval_f1_macro"])
        result["test_top2_accuracy"] = float(test_metrics["eval_top2_accuracy"])

    if save_model:
        # free memory before saving (this machine runs low on RAM)
        import gc

        del trainer.optimizer, trainer.lr_scheduler
        trainer.optimizer = None
        trainer.lr_scheduler = None
        model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        os.makedirs(out_dir, exist_ok=True)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune DistilBERT on ticket routing.")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--subset", type=int, default=None,
                   help="Use only N training examples (for a fast smoke test).")
    p.add_argument("--out", type=str, default="models/run")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--test", action="store_true", help="Also evaluate on the test split.")
    p.add_argument("--class_weights", action="store_true",
                   help="Weight the loss by inverse class frequency (helps macro-F1).")
    args = p.parse_args()

    metrics = train_model(
        epochs=args.epochs,
        lr=args.lr,
        subset=args.subset,
        out_dir=args.out,
        batch_size=args.batch_size,
        max_length=args.max_length,
        evaluate_test=args.test,
        class_weights=args.class_weights,
    )
    print("\n" + "=" * 50)
    print("RUN FINISHED - validation metrics")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:16s}: {v}")
    print(f"\nModel saved to: {args.out}")


if __name__ == "__main__":
    bigstack.run(main)
