"""
SSAQS feature builder, v2. Same output as build_ssaqs_features.py plus two switches:

  --hrv-local-tz-hours H   heart-rate timestamps have no 'Z' suffix, so they may be local time.
                           If they are local time at UTC+H, pass H (e.g. -6 for UTC-6).
                           Default 0 keeps the original behaviour.
  --drop-no-hrv            keep only responses with heart-rate data in the 24 h before them.

The script prints how many responses have heart-rate data, so you can see whether the
timezone shift closes the gap before running the methodology.

Usage:
    python build_ssaqs_features_v2.py --src "D:\\THESIS PROJECT\\SSAQS_raw" --threshold 50 \
        --hrv-local-tz-hours -6 --out "D:\\THESIS PROJECT\\SSAQS\\ssaqs_features_v2.csv"
"""
import argparse
import os

import pandas as pd

WINDOW = pd.Timedelta(hours=24)
SLEEP_LOOKBACK = pd.Timedelta(hours=48)


def read_csv(folder, name):
    p = os.path.join(folder, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def to_utc(s):
    return pd.to_datetime(s, utc=True, errors="coerce")


def load_timed(folder, name, shift_hours=0.0):
    df = read_csv(folder, name)
    if df is None:
        return None
    df["time"] = to_utc(df["timestamp"]) - pd.Timedelta(hours=shift_hours)
    return df.dropna(subset=["time"]).sort_values("time")


def build_participant(folder, pid, threshold, hrv_shift, drop_no_hrv):
    dq = read_csv(folder, "daily_questions.csv")
    if dq is None:
        return [], 0
    dq = dq.copy()
    dq["time"] = pd.to_datetime(dq["timeStampSent"], unit="s", utc=True, errors="coerce")
    dq = dq[~((dq["stress"] == 0) & (dq["anxiety"] == 0))]

    hrv = load_timed(folder, "hrv.csv", hrv_shift)
    oxy = load_timed(folder, "oxygen.csv")
    if oxy is not None:
        oxy = oxy[oxy["value"] > 50]  # 50.0 is a placeholder value in the export
    steps = load_timed(folder, "steps.csv")
    act = load_timed(folder, "activity_level.csv")
    sleep = load_timed(folder, "sleep.csv")

    rows, dropped = [], 0
    for i, r in dq.iterrows():
        t = r["time"]
        if pd.isna(t):
            continue
        t0 = t - WINDOW
        feat = {}

        if hrv is not None:
            w = hrv[(hrv["time"] >= t0) & (hrv["time"] < t)]
            feat["hrv_n_samples"] = len(w)
            if len(w):
                for c in ["rmssd", "low_frequency", "high_frequency", "coverage"]:
                    feat[f"hrv_{c}_mean"] = w[c].mean()
            if len(w) >= 2:
                feat["hrv_rmssd_std"] = w["rmssd"].std()

        if drop_no_hrv and feat.get("hrv_n_samples", 0) == 0:
            dropped += 1
            continue

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
                for lvl, sh in w["level"].value_counts(normalize=True).items():
                    feat[f"activity_share_{str(lvl).lower()}"] = sh

        if sleep is not None:
            w = sleep[(sleep["time"] < t) & (sleep["time"] >= t - SLEEP_LOOKBACK)]
            if len(w):
                feat["sleep_score"] = w.iloc[-1]["overall_score"]
                feat["sleep_deep_min"] = w.iloc[-1]["deep_sleep_in_minutes"]

        rows.append({
            "file": f"{pid}_{i}",
            "speaker_id": pid,
            "label": int(r["stress"] >= threshold),
            **feat,
        })
    return rows, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--threshold", type=float, default=50.0)
    ap.add_argument("--hrv-local-tz-hours", type=float, default=0.0)
    ap.add_argument("--drop-no-hrv", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    folders = sorted(d for d in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, d)))
    if not folders:
        raise SystemExit(f"no participant folders under {args.src}")

    all_rows, total_dropped = [], 0
    for pid in folders:
        rows, dropped = build_participant(os.path.join(args.src, pid), pid, args.threshold,
                                          args.hrv_local_tz_hours, args.drop_no_hrv)
        all_rows.extend(rows)
        total_dropped += dropped

    out = pd.DataFrame(all_rows).dropna(axis=1, how="all")
    out.to_csv(args.out, index=False)

    hrv_share = (out["hrv_n_samples"] > 0).mean() if "hrv_n_samples" in out else float("nan")
    print(f"responses            : {len(out)}")
    print(f"speakers             : {out['speaker_id'].nunique()}")
    print(f"features             : {out.shape[1] - 3}")
    print(f"responses with HRV   : {hrv_share:.0%}")
    if args.drop_no_hrv:
        print(f"dropped (no HRV)     : {total_dropped}")
    print(f"label 1 (stress >= {args.threshold:g}): {int(out['label'].sum())}")
    print(f"label 0              : {int((out['label'] == 0).sum())}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()