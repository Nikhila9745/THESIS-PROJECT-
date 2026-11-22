"""
Build one row per daily questionnaire response for the SSAQS wearable dataset.

Input layout (what the listing showed):
    SSAQS_raw/<participant>/daily_questions.csv   (label source: self-reported stress 0-100)
    SSAQS_raw/<participant>/hrv.csv               (5-min HRV: rmssd, low_frequency, high_frequency, coverage)
    SSAQS_raw/<participant>/sleep.csv             (overall_score, deep_sleep_in_minutes)
    SSAQS_raw/<participant>/steps.csv             (per-minute steps)
    SSAQS_raw/<participant>/activity_level.csv    (per-minute activity level)
    SSAQS_raw/<participant>/oxygen.csv            (SpO2 value)

Not used as features:
    stress.csv   - device-computed STRESS_SCORE, probably derived from HRV/sleep, so using it
                   would leak the signal we are trying to predict.
    anxiety      - a second self-report, excluded so the task stays "predict stress".

Each EMA response gets features from the 24 hours before it, plus the last sleep record
before it. The label is 1 if stress >= --threshold, else 0.

Usage:
    python build_ssaqs_features.py --src "D:\\THESIS PROJECT\\SSAQS_raw" --threshold 50 \
        --out "D:\\THESIS PROJECT\\SSAQS\\ssaqs_features.csv"
"""
import argparse
import os

import numpy as np
import pandas as pd

WINDOW = pd.Timedelta(hours=24)
SLEEP_LOOKBACK = pd.Timedelta(hours=48)


def read_csv(folder, name):
    p = os.path.join(folder, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def to_utc(s):
    return pd.to_datetime(s, utc=True, errors="coerce")


def window_mean(ts_series, values, t_end, t_start):
    m = (ts_series >= t_start) & (ts_series < t_end)
    return values[m]


def build_participant(folder, pid, threshold):
    dq = read_csv(folder, "daily_questions.csv")
    if dq is None:
        return []
    dq = dq.copy()
    dq["time"] = pd.to_datetime(dq["timeStampSent"], unit="s", utc=True, errors="coerce")
    # responses where both answers are 0 are treated as empty submissions and dropped
    dq = dq[~((dq["stress"] == 0) & (dq["anxiety"] == 0))]

    hrv = read_csv(folder, "hrv.csv")
    if hrv is not None:
        hrv["time"] = to_utc(hrv["timestamp"])
        hrv = hrv.dropna(subset=["time"]).sort_values("time")
    oxy = read_csv(folder, "oxygen.csv")
    if oxy is not None:
        oxy["time"] = to_utc(oxy["timestamp"])
        oxy = oxy.dropna(subset=["time"]).sort_values("time")
        oxy = oxy[oxy["value"] > 50]  # 50.0 appears as a placeholder value in the export
    steps = read_csv(folder, "steps.csv")
    if steps is not None:
        steps["time"] = to_utc(steps["timestamp"])
        steps = steps.dropna(subset=["time"]).sort_values("time")
    act = read_csv(folder, "activity_level.csv")
    if act is not None:
        act["time"] = to_utc(act["timestamp"])
        act = act.dropna(subset=["time"]).sort_values("time")
    sleep = read_csv(folder, "sleep.csv")
    if sleep is not None:
        sleep["time"] = to_utc(sleep["timestamp"])
        sleep = sleep.dropna(subset=["time"]).sort_values("time")

    rows = []
    for i, r in dq.iterrows():
        t = r["time"]
        if pd.isna(t):
            continue
        t0 = t - WINDOW
        feat = {}

        if hrv is not None:
            w = hrv[(hrv["time"] >= t0) & (hrv["time"] < t)]
            for c in ["rmssd", "low_frequency", "high_frequency", "coverage"]:
                if len(w):
                    feat[f"hrv_{c}_mean"] = w[c].mean()
            if len(w) >= 2:
                feat["hrv_rmssd_std"] = w["rmssd"].std()
            feat["hrv_n_samples"] = len(w)

        if oxy is not None:
            w = oxy[(oxy["time"] >= t0) & (oxy["time"] < t)]
            if len(w):
                feat["spo2_mean"] = w["value"].mean()
                feat["spo2_min"] = w["value"].min()

        if steps is not None:
            w = steps[(steps["time"] >= t0) & (steps["time"] < t)]
            feat["steps_24h"] = w["steps"].sum() if len(w) else 0

        if act is not None:
            w = act[(act["time"] >= t0) & (act["time"] < t)]
            if len(w):
                shares = w["level"].value_counts(normalize=True)
                for lvl, sh in shares.items():
                    feat[f"activity_share_{str(lvl).lower()}"] = sh

        if sleep is not None:
            w = sleep[(sleep["time"] < t) & (sleep["time"] >= t - SLEEP_LOOKBACK)]
            if len(w):
                last = w.iloc[-1]
                feat["sleep_score"] = last["overall_score"]
                feat["sleep_deep_min"] = last["deep_sleep_in_minutes"]

        rows.append({
            "file": f"{pid}_{i}",
            "speaker_id": pid,
            "label": int(r["stress"] >= threshold),
            "stress_raw": r["stress"],
            **feat,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder containing one sub-folder per participant")
    ap.add_argument("--threshold", type=float, default=50.0,
                    help="stress >= threshold is label 1 (positive class)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    folders = sorted(d for d in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, d)))
    if not folders:
        raise SystemExit(f"no participant folders under {args.src}")

    all_rows = []
    for pid in folders:
        rows = build_participant(os.path.join(args.src, pid), pid, args.threshold)
        all_rows.extend(rows)
        print(f"participant {pid}: {len(rows)} responses")

    out = pd.DataFrame(all_rows)
    # the pipeline treats every numeric column as a feature, so drop the raw label copy
    out = out.drop(columns=["stress_raw"])
    out = out.dropna(axis=1, how="all")
    out.to_csv(args.out, index=False)

    print(f"\nresponses : {len(out)}")
    print(f"speakers  : {out['speaker_id'].nunique()}")
    print(f"features  : {out.shape[1] - 3}")
    print(f"label 1 (stress >= {args.threshold:g}): {int(out['label'].sum())}")
    print(f"label 0               : {int((out['label'] == 0).sum())}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()