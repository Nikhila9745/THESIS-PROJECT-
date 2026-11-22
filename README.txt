STRESS METHODS KIT
==================

Run everything from the Ubuntu terminal with the sparkenv active:
  source ~/sparkenv/bin/activate
  cd "/mnt/c/Users/JP/THESIS PROJECT"
  pip install pyarrow        (speeds up Spark-to-pandas conversion; optional)

1. METHODS SUITE (one run per dataset size)
   python3 methods_suite.py --csv "/mnt/d/THESIS PROJECT/Stress_Data_5GB.csv"  --out "/mnt/c/Users/JP/THESIS PROJECT/methods_5GB"
   python3 methods_suite.py --csv "/mnt/d/THESIS PROJECT/Stress_Data_10GB.csv" --out "/mnt/c/Users/JP/THESIS PROJECT/methods_10GB"
   python3 methods_suite.py --csv "/mnt/c/Users/JP/THESIS PROJECT/Stress_Data_15GB.csv" --out "/mnt/c/Users/JP/THESIS PROJECT/methods_15GB"
   python3 methods_suite.py --csv "/mnt/c/Users/JP/THESIS PROJECT/Stress_Data_20GB.csv" --out "/mnt/c/Users/JP/THESIS PROJECT/methods_20GB"
   Options: --seeds 5 (default), --bootstrap 1000 (default), --per-class 10000 (default).
   Each run writes: repeated_seed_metrics.csv, repeated_summary.csv (mean and SD over seeds),
   bootstrap_ci_seed42.csv, mcnemar_tests_seed42.csv, learning_curve.csv,
   ablation_seed42.csv, label_derivation_check.csv, run_summary.json.

2. STRONG SCALING (same file, different core counts; 5 GB example)
   python3 strong_scaling.py --csv "/mnt/d/THESIS PROJECT/Stress_Data_5GB.csv" --cores 1   --out "/mnt/c/Users/JP/THESIS PROJECT/scaling_5GB"
   python3 strong_scaling.py --csv "/mnt/d/THESIS PROJECT/Stress_Data_5GB.csv" --cores 2   --out "/mnt/c/Users/JP/THESIS PROJECT/scaling_5GB"
   python3 strong_scaling.py --csv "/mnt/d/THESIS PROJECT/Stress_Data_5GB.csv" --cores 4   --out "/mnt/c/Users/JP/THESIS PROJECT/scaling_5GB"
   python3 strong_scaling.py --csv "/mnt/d/THESIS PROJECT/Stress_Data_5GB.csv" --cores all --out "/mnt/c/Users/JP/THESIS PROJECT/scaling_5GB"
   Each run appends one row to strong_scaling.csv. Use the same file for every core count.

3. WEBSITE (numeric-attribute model, browser prediction)
   python3 export_numeric_model.py --csv "/mnt/d/THESIS PROJECT/Stress_Data_5GB.csv" --out "/mnt/c/Users/JP/THESIS PROJECT/website/model.json"
   The script prints two probability lists. They must match. This writes website/model_data.js.
   Then open "C:\Users\JP\THESIS PROJECT\website\index.html" in a browser.
   The page is a research demonstration. It is not a diagnostic tool.

LIMITS
- The methods suite and strong scaling were tested only on the synthetic test file, not on your datasets.
- The website model uses numeric attributes only, so it is less accurate than the full pipeline.
- Naive Bayes uses min-max scaling with values clipped to [0, 1], because Naive Bayes needs non-negative inputs.
- SHAP is not included; it needs the shap package and a sample exported to pandas. Add it later if needed.
