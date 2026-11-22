"""
LINE-BY-LINE EXPLAINED VERSION of: Build ssaqs features v2.py
The code is unchanged. Every line or block has a comment above it saying what it does.

WHAT THIS SCRIPT DOES IN ONE PARAGRAPH
For every participant folder (SSAQS_raw/<participant>/), it reads the daily stress/anxiety
questionnaire answers. For each answer it looks at the wearable data (heart-rate variability,
oxygen, steps, activity, sleep) from the 24 hours BEFORE the answer, summarises that data into
numbers (features), and attaches a 0/1 label (1 = stress score >= threshold). The result is one
CSV with one row per questionnaire answer, ready for machine learning.
"""
# ---------------------------------------------------------------------------
# Original docstring (explains the two new switches added in v2)
# ---------------------------------------------------------------------------
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
# Command-line tools: argparse reads the --options you type; os handles folders and file paths.
import argparse
import os

# pandas is the table library used for all the data handling below.
import pandas as pd

# WINDOW: how far back from each questionnaire answer we collect wearable data (24 hours).
WINDOW = pd.Timedelta(hours=24)
# SLEEP_LOOKBACK: sleep summaries are only used if they are within the previous 48 hours.
SLEEP_LOOKBACK = pd.Timedelta(hours=48)


# Helper: read one CSV from a participant folder, or give back None if the file is not there.
def read_csv(folder, name):
    # Join the folder path and the file name into one full path.
    p = os.path.join(folder, name)
    # If the file exists read it into a table, otherwise return None (not every participant has every sensor).
    return pd.read_csv(p) if os.path.exists(p) else None


# Helper: turn text timestamps into timezone-aware UTC datetimes.
def to_utc(s):
    # utc=True marks the times as UTC. errors="coerce" turns unreadable values into NaT (missing time) instead of crashing.
    return pd.to_datetime(s, utc=True, errors="coerce")


# Helper: load a sensor CSV, give it a clean "time" column, and sort it by time.
# shift_hours is the timezone correction (only used for the HRV file).
def load_timed(folder, name, shift_hours=0.0):
    # Read the file (None if missing).
    df = read_csv(folder, name)
    # No file -> no data for this sensor; the caller handles None.
    if df is None:
        return None
    # Parse the "timestamp" column as UTC, then subtract shift_hours.
    # If the device wrote local time at UTC+H, then real UTC = local - H. For UTC-6 (H = -6) this ADDS 6 hours.
    df["time"] = to_utc(df["timestamp"]) - pd.Timedelta(hours=shift_hours)
    # Drop rows whose time could not be parsed, and sort oldest -> newest (needed for the "latest sleep" logic below).
    return df.dropna(subset=["time"]).sort_values("time")


# Main per-participant function. Returns (list of feature rows, number of rows dropped for having no HRV).
def build_participant(folder, pid, threshold, hrv_shift, drop_no_hrv):
    # Load the questionnaire answers, which are the "prediction points".
    dq = read_csv(folder, "daily_questions.csv")
    # No questionnaire file -> nothing to build for this participant.
    if dq is None:
        return [], 0
    # Work on a copy so we never modify the original table by accident.
    dq = dq.copy()
    # timeStampSent is Unix seconds; convert it to a UTC datetime in a new "time" column.
    dq["time"] = pd.to_datetime(dq["timeStampSent"], unit="s", utc=True, errors="coerce")
    # Remove answers where BOTH stress and anxiety are 0 (almost certainly a default/unanswered slider, not a real report).
    dq = dq[~((dq["stress"] == 0) & (dq["anxiety"] == 0))]

    # Load heart-rate variability data, applying the timezone shift.
    hrv = load_timed(folder, "hrv.csv", hrv_shift)
    # Load blood-oxygen data (no shift).
    oxy = load_timed(folder, "oxygen.csv")
    # Only if the oxygen file exists:
    if oxy is not None:
        # Keep readings strictly above 50. The export uses exactly 50.0 as a "no reading" placeholder, so it is removed.
        oxy = oxy[oxy["value"] > 50]  # 50.0 is a placeholder value in the export
    # Load step counts.
    steps = load_timed(folder, "steps.csv")
    # Load activity-level records (e.g. sedentary / active).
    act = load_timed(folder, "activity_level.csv")
    # Load nightly sleep summaries.
    sleep = load_timed(folder, "sleep.csv")

    # rows collects one dictionary per questionnaire answer; dropped counts answers skipped for missing HRV.
    rows, dropped = [], 0
    # Go through every questionnaire answer. i is the row label, r is the row.
    for i, r in dq.iterrows():
        # t = the moment the question was sent.
        t = r["time"]
        # Skip answers with an unreadable time.
        if pd.isna(t):
            continue
        # t0 = start of the 24-hour window before the answer.
        t0 = t - WINDOW
        # feat will hold all the numbers computed for this answer.
        feat = {}

        # ---- Heart-rate variability features ----
        if hrv is not None:
            # Select HRV samples with t0 <= time < t (strictly BEFORE the answer, so no future information leaks in).
            w = hrv[(hrv["time"] >= t0) & (hrv["time"] < t)]
            # Count how many samples fell in the window (also used later to measure HRV coverage).
            feat["hrv_n_samples"] = len(w)
            # If there is at least one sample:
            if len(w):
                # Average each of these four HRV columns over the window.
                for c in ["rmssd", "low_frequency", "high_frequency", "coverage"]:
                    feat[f"hrv_{c}_mean"] = w[c].mean()
            # A standard deviation needs at least two points.
            if len(w) >= 2:
                # Variability of rmssd within the window.
                feat["hrv_rmssd_std"] = w["rmssd"].std()

        # --drop-no-hrv: skip this answer entirely if there was no HRV data in the window.
        if drop_no_hrv and feat.get("hrv_n_samples", 0) == 0:
            # Count it as dropped, and move to the next answer.
            dropped += 1
            continue

        # ---- Oxygen features ----
        if oxy is not None:
            # Oxygen readings in the same 24-hour window.
            w = oxy[(oxy["time"] >= t0) & (oxy["time"] < t)]
            if len(w):
                # Average and lowest oxygen level.
                feat["spo2_mean"] = w["value"].mean()
                feat["spo2_min"] = w["value"].min()

        # ---- Step features ----
        if steps is not None:
            # Step records in the window.
            w = steps[(steps["time"] >= t0) & (steps["time"] < t)]
            # Total steps; if there are no records it is set to 0 (note: "no data" becomes "0 steps").
            feat["steps_24h"] = w["steps"].sum() if len(w) else 0

        # ---- Activity-level features ----
        if act is not None:
            # Activity records in the window.
            w = act[(act["time"] >= t0) & (act["time"] < t)]
            if len(w):
                # For each activity level, compute the SHARE of records (0-1) and store it as activity_share_<level>.
                for lvl, sh in w["level"].value_counts(normalize=True).items():
                    feat[f"activity_share_{str(lvl).lower()}"] = sh

        # ---- Sleep features ----
        if sleep is not None:
            # Sleep records before the answer but not older than 48 hours.
            w = sleep[(sleep["time"] < t) & (sleep["time"] >= t - SLEEP_LOOKBACK)]
            if len(w):
                # The table is sorted by time, so the last row (iloc[-1]) is the most recent night.
                feat["sleep_score"] = w.iloc[-1]["overall_score"]
                feat["sleep_deep_min"] = w.iloc[-1]["deep_sleep_in_minutes"]

        # Build the final row for this answer.
        rows.append({
            # Unique id: participant id + row label.
            "file": f"{pid}_{i}",
            # Which participant it came from (needed to split train/test by person).
            "speaker_id": pid,
            # Binary label: 1 if the stress score is at or above the threshold, else 0.
            "label": int(r["stress"] >= threshold),
            # Add all computed features to the same row.
            **feat,
        })
    # Hand back the rows and the number dropped.
    return rows, dropped


# Entry point: reads the command-line options and runs everything.
def main():
    # Create the option parser.
    ap = argparse.ArgumentParser()
    # Required: folder containing one sub-folder per participant.
    ap.add_argument("--src", required=True)
    # Stress score cut-off for label 1 (default 50).
    ap.add_argument("--threshold", type=float, default=50.0)
    # Timezone shift for HRV timestamps (default 0 = treat them as UTC).
    ap.add_argument("--hrv-local-tz-hours", type=float, default=0.0)
    # On/off switch: keep only answers that have HRV data.
    ap.add_argument("--drop-no-hrv", action="store_true")
    # Required: where to save the output CSV.
    ap.add_argument("--out", required=True)
    # Read the options you typed.
    args = ap.parse_args()

    # List the participant folders (ignore stray files), in alphabetical order.
    folders = sorted(d for d in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, d)))
    # Stop with a clear message if there are none.
    if not folders:
        raise SystemExit(f"no participant folders under {args.src}")

    # all_rows collects rows from everyone; total_dropped sums the dropped counts.
    all_rows, total_dropped = [], 0
    # Process each participant, using the folder name as the participant id.
    for pid in folders:
        rows, dropped = build_participant(os.path.join(args.src, pid), pid, args.threshold,
                                          args.hrv_local_tz_hours, args.drop_no_hrv)
        # Add this participant's rows to the big list.
        all_rows.extend(rows)
        # Add to the dropped total.
        total_dropped += dropped

    # Turn the list of dictionaries into one table; drop any column that is empty for everybody.
    out = pd.DataFrame(all_rows).dropna(axis=1, how="all")
    # Save to CSV without the row-number column.
    out.to_csv(args.out, index=False)

    # Share of answers that have at least one HRV sample (NaN if the HRV column does not exist).
    hrv_share = (out["hrv_n_samples"] > 0).mean() if "hrv_n_samples" in out else float("nan")
    # Summary printout: number of answers.
    print(f"responses            : {len(out)}")
    # Number of distinct participants.
    print(f"speakers             : {out['speaker_id'].nunique()}")
    # Number of feature columns = all columns minus file, speaker_id, label.
    print(f"features             : {out.shape[1] - 3}")
    # How many answers have HRV data (use this to judge whether the timezone shift helped).
    print(f"responses with HRV   : {hrv_share:.0%}")
    # Only show the dropped count if the drop switch was used.
    if args.drop_no_hrv:
        print(f"dropped (no HRV)     : {total_dropped}")
    # Class balance: how many high-stress answers.
    print(f"label 1 (stress >= {args.threshold:g}): {int(out['label'].sum())}")
    # How many low-stress answers.
    print(f"label 0              : {int((out['label'] == 0).sum())}")
    # Confirm where the file was saved.
    print(f"saved -> {args.out}")


# Run main() only when the file is executed directly (not when imported).
if __name__ == "__main__":
    main()
