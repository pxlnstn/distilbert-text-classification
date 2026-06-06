"""Print dataset properties: topic, classes, sample counts, class balance.

    python src/explore_data.py
"""

from collections import Counter

from data_prep import HF_DATASET, ID2LABEL, LABELS, NUM_LABELS, load_prepared_datasets


def main() -> None:
    dd = load_prepared_datasets()
    total = sum(split.num_rows for split in dd.values())

    print("=" * 64)
    print("DATASET PROPERTIES")
    print("=" * 64)
    print(f"Source (HuggingFace) : {HF_DATASET}")
    print("Topic                : Enterprise customer-support / IT-incident ticket")
    print("                       routing - classify a ticket into the department")
    print("                       ('queue') that should handle it.")
    print("Language used        : English only")
    print(f"Number of classes    : {NUM_LABELS}")
    print("Input feature        : 1 engineered text feature (subject + body)")
    print("Target               : queue (department)")
    print(f"Total samples (used) : {total}")
    print("Split sizes          :")
    for name, split in dd.items():
        print(f"    {name:12s}: {split.num_rows:6d}")

    print("-" * 64)
    print("Class distribution (training split):")
    counts = Counter(dd["train"]["label"])
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    train_total = dd["train"].num_rows
    for label_id, n in ordered:
        pct = 100.0 * n / train_total
        bar = "#" * int(pct)
        print(f"  {ID2LABEL[label_id]:32s} {n:6d} ({pct:4.1f}%) {bar}")

    most = ordered[0][1]
    least = ordered[-1][1]
    print("-" * 64)
    print(f"Class imbalance ratio (largest/smallest): {most/least:.1f}x")
    print("=> The data is imbalanced, so we report macro-F1 (treats every class")
    print("   equally) alongside plain accuracy.")
    print("=" * 64)


if __name__ == "__main__":
    main()
