"""Build a small, simple analog-clock reading set from TickTockVQA (Jaeha-Choi/TickTockVQA, CVPR'26 Findings).

Simple cases = upright clock (transformation Normal), wall or alarm/desk clock (usually face-on and
close), Arabic numerals. Only CC BY 4.0 rows (commercial use + redistribution allowed).
Train images come from the dataset's train split, eval images from its test split (no overlap).

Output:
    data/clock/{train,eval}/clock_XXX.jpg
    data/clock/labels/{train,eval}.jsonl
        {"image": "train/clock_000.jpg", "hour": 1-12, "minute": 0-59, "time": "hh:mm", ...source/license}
The answer is 12-hour hh:mm (a clock face shows no AM/PM); the dataset's AM/PM is kept as `ampm`.

    python tools/prepare_clock_data.py                 # 100 train / 20 eval
    python tools/prepare_clock_data.py --n_train 500   # more
"""
import argparse
import io
import json
import os

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "Jaeha-Choi/TickTockVQA"
# shards that hold CC BY 4.0 rows (train-00004 is SBU, non-commercial only)
SHARDS = {"train": [f"train-0000{i}-of-00005.parquet" for i in range(4)],
          "test": ["test-00001-of-00004.parquet"]}
CLOCK_TYPES = ["Wall clocks", "Alarm/Desk clocks"]
LICENSE = "Creative Commons Attribution 4.0 International"
KEEP_COLS = ["clock_type", "design", "environment", "ampm", "source", "source_url", "license",
             "license_type", "license_url", "attribution_required", "image_name", "image_hash"]


def load(split):
    frames = []
    for f in SHARDS[split]:
        p = hf_hub_download(REPO, f, repo_type="dataset")
        frames.append(pq.read_table(p).to_pandas())
    return pd.concat(frames, ignore_index=True)


def simple(df):
    m = ((df.transformation == "Normal") & df.clock_type.isin(CLOCK_TYPES)
         & df.design.apply(lambda d: "Arabic" in list(d)) & (df.license_type == LICENSE)
         & df.redistribution_allowed & df.commercial_use
         & df.hour.between(0, 12) & df.minute.between(0, 59))
    return df[m].drop_duplicates("image_hash")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/clock")
    ap.add_argument("--n_train", type=int, default=100)
    ap.add_argument("--n_eval", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop_1010", type=lambda v: v.lower() in ("1", "true", "yes"), default=True,
                    help="drop train clocks within 5 min of 10:10 (product-photo pose; the base model's "
                         "default guess, so keeping them rewards the prior instead of reading the hands)")
    args = ap.parse_args()

    train_pool, test_pool = simple(load("train")), simple(load("test"))
    test_pool = test_pool[~test_pool.image_hash.isin(set(train_pool.image_hash))]
    if args.drop_1010:
        near = (train_pool.hour % 12 == 10) & train_pool.minute.between(5, 15)
        print(f"dropping {int(near.sum())} train candidates near 10:10")
        train_pool = train_pool[~near]
    print(f"simple CC-BY candidates: train {len(train_pool)}, test {len(test_pool)}")

    os.makedirs(f"{args.out}/labels", exist_ok=True)
    for split, pool, n in (("train", train_pool, args.n_train), ("eval", test_pool, args.n_eval)):
        sel = pool.sample(n=n, random_state=args.seed).reset_index(drop=True)
        os.makedirs(f"{args.out}/{split}", exist_ok=True)
        with open(f"{args.out}/labels/{split}.jsonl", "w") as f:
            for i, r in sel.iterrows():
                img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
                rel = f"{split}/clock_{i:03d}.jpg"
                img.save(f"{args.out}/{rel}", quality=95)
                hour = 12 if int(r.hour) == 0 else int(r.hour)
                row = {"image": rel, "width": img.width, "height": img.height,
                       "hour": hour, "minute": int(r.minute), "time": f"{hour:02d}:{int(r.minute):02d}",
                       **{c: (list(r[c]) if c == "design" else (None if pd.isna(r[c]) else r[c])) for c in KEEP_COLS}}
                row["attribution_required"] = bool(row["attribution_required"])
                f.write(json.dumps(row, default=str) + "\n")
        print(f"{split}: {n} images -> {args.out}/{split}/, labels -> {args.out}/labels/{split}.jsonl")
        print("  clock types", sel.clock_type.value_counts().to_dict(), "| sources", sel.source.value_counts().to_dict())


if __name__ == "__main__":
    main()
