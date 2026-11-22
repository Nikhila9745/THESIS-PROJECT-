"""
Prepare the SWELL Kaggle download for audio_stress_pipeline.py.

Step 1 - inspect (run once, no flags needed beyond --src):
    python prepare_swell.py --src "C:\\path\\to\\kagglehub\\download\\folder"

Step 2 - build the features CSV (fill in the column names printed in step 1):
    python prepare_swell.py --src "C:\\path\\..." --label-col "<label column>" \
        --speaker-col "<participant column>" --positive "<stress label value>" \
        --out swell_features.csv

Output columns: file, speaker_id, label (0/1 after --positive), then numeric features.
Then run:
    python audio_stress_pipeline.py --features swell_features.csv --positive 1 --out swell_results
"""
import argparse
import glob
import os

import pandas as pd


def find_csvs(src):
    paths = glob.glob(os.path.join(src, "**", "*.csv"), recursive=True)
    if not paths:
        raise SystemExit(f"no .csv files found under {src}")
    return paths


def inspect(paths):
    for p in paths:
        df = pd.read_csv(p, nrows=5)
        print(f"\n== {p}")
        print("columns:", list(df.columns))
        print(df.head(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder returned by kagglehub")
    ap.add_argument("--file", default=None, help="CSV to use (default: first CSV found)")
    ap.add_argument("--label-col")
    ap.add_argument("--speaker-col")
    ap.add_argument("--positive", help="label value treated as stress (label 1)")
    ap.add_argument("--out", default="swell_features.csv")
    args = ap.parse_args()

    paths = find_csvs(args.src)
    if not args.label_col:
        inspect(paths)
        print("\nRe-run with --label-col, --speaker-col and --positive once you see the columns.")
        return

    src_csv = args.file or paths[0]
    df = pd.read_csv(src_csv)
    for c in (args.label_col, args.speaker_col):
        if c not in df.columns:
            raise SystemExit(f"column '{c}' not found; available: {list(df.columns)}")

    out = pd.DataFrame()
    out["file"] = [f"row_{i:06d}" for i in range(len(df))]
    out["speaker_id"] = df[args.speaker_col].astype(str)
    out["label"] = (df[args.label_col].astype(str) == str(args.positive)).astype(int)

    numeric = df.select_dtypes(include="number").drop(columns=[args.label_col], errors="ignore")
    # drop columns that leak the label or are identifiers
    drop_like = [c for c in numeric.columns if "stress" in c.lower() or "label" in c.lower()
                 or c.lower() in {"id", "index", "unnamed: 0"}]
    numeric = numeric.drop(columns=drop_like, errors="ignore")
    out = pd.concat([out, numeric.reset_index(drop=True)], axis=1)
    out = out.dropna(thresh=max(3, int(0.5 * out.shape[1])))  # drop rows that are mostly empty

    out.to_csv(args.out, index=False)
    print(f"source        : {src_csv}")
    print(f"rows kept     : {len(out)}")
    print(f"speakers      : {out['speaker_id'].nunique()}")
    print(f"label counts  :\n{out['label'].value_counts().to_string()}")
    print(f"dropped cols  : {drop_like}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()