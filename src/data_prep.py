"""
Data preparation for the customer-support ticket classifier.

Dataset : Tobi-Bueck/customer-support-tickets (HuggingFace)
Task    : route a support ticket to the correct DEPARTMENT (the `queue` field).
We keep ENGLISH rows only, build one text field from subject + body, map the
10 queue names to integer label ids, and create a stratified train/val/test split.

This module is imported by explore_data.py, train.py and run_experiments.py.
Running it directly will build + cache the split and print a short summary.
"""

from __future__ import annotations

from datasets import ClassLabel, Dataset, DatasetDict, load_dataset

HF_DATASET = "Tobi-Bueck/customer-support-tickets"

# alphabetical order -> label ids stay the same across runs
LABELS = [
    "Billing and Payments",
    "Customer Service",
    "General Inquiry",
    "Human Resources",
    "IT Support",
    "Product Support",
    "Returns and Exchanges",
    "Sales and Pre-Sales",
    "Service Outages and Maintenance",
    "Technical Support",
]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}
NUM_LABELS = len(LABELS)


def _build_text(subject: str | None, body: str | None) -> str:
    """Combine the e-mail subject and body into a single input string."""
    subject = (subject or "").strip()
    body = (body or "").strip()
    if subject and body:
        return f"{subject}\n\n{body}"
    return subject or body


def load_prepared_datasets(
    subset: int | None = None,
    seed: int = 42,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
) -> DatasetDict:
    """
    Returns a DatasetDict with 'train', 'validation' and 'test' splits.
    Each example has two columns: 'text' (str) and 'label' (int 0-9).

    subset: if given, keep only this many TRAINING examples (stratified-ish via
            shuffle) for fast smoke tests / quick experiment runs. Validation and
            test sets are always kept full so metrics stay comparable.
    """
    raw = load_dataset(HF_DATASET)["train"]

    # english rows with a valid queue and some text
    raw = raw.filter(
        lambda r: r["language"] == "en"
        and r["queue"] in LABEL2ID
        and _build_text(r["subject"], r["body"]) != ""
    )

    def to_text_label(r):
        return {"text": _build_text(r["subject"], r["body"]), "label": LABEL2ID[r["queue"]]}

    cols_to_drop = [c for c in raw.column_names if c not in ("text", "label")]
    ds = raw.map(to_text_label, remove_columns=cols_to_drop)

    # ClassLabel lets us stratify the split
    ds = ds.cast_column("label", ClassLabel(names=LABELS))

    # split off val+test, then halve it into val and test
    holdout = val_frac + test_frac
    split1 = ds.train_test_split(test_size=holdout, seed=seed, stratify_by_column="label")
    train_ds = split1["train"]
    rest = split1["test"]
    split2 = rest.train_test_split(
        test_size=test_frac / holdout, seed=seed, stratify_by_column="label"
    )
    val_ds = split2["train"]
    test_ds = split2["test"]

    # smaller train set for quick runs
    if subset is not None and subset < train_ds.num_rows:
        train_ds = train_ds.shuffle(seed=seed).select(range(subset))

    return DatasetDict(train=train_ds, validation=val_ds, test=test_ds)


if __name__ == "__main__":
    dd = load_prepared_datasets()
    print("Prepared splits:")
    for name, split in dd.items():
        print(f"  {name:12s}: {split.num_rows:6d} examples")
    print("\nExample training row:")
    ex = dd["train"][0]
    print("  label:", ex["label"], "->", ID2LABEL[ex["label"]])
    print("  text :", ex["text"][:160].replace("\n", " "), "...")
