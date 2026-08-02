"""
================================================================================
VALIDATION SCRIPT — Synthetic vs. Real Anchor Distributions
================================================================================

Purpose:
  Verify that the generated synthetic dataset reproduces the marginal
  distributions defined by its anchors. This is the first-line check.

  Full TSTR validation against the 50 real Slave Island records is
  performed separately — this script prepares the synthetic side and
  provides a slot for the real 50 to be loaded.

Tests:
  1. Marginal fidelity — synthetic distribution vs anchor proportions
     (chi-square goodness of fit).
  2. Slave Island slice — verify Slave Island subset also matches anchors.
  3. Placeholder: real 50 loader for TSTR comparison.
================================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

# Resolve paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_CSV_PATH = os.path.join(
    PROJECT_ROOT, "data", "raw",
    "synthetic_cma_accidents_2022_2023.csv"
)

INPUT_CSV = INPUT_CSV_PATH

# Anchor proportions to test against
ANCHOR_SEVERITY = {
    "Fatal":       2583 / 22153,
    "Serious":     5276 / 22153,
    "Minor":       10067 / 22153,
    "Damage-only": 4227 / 22153,
}

ANCHOR_TIME_FATAL = {
    "06:00-11:59": 0.269, "12:00-17:59": 0.247,
    "18:00-23:59": 0.392, "00:00-05:59": 0.093,
}


def time_to_band_fatal(t):
    hh = int(t.split(":")[0])
    if 6 <= hh < 12:  return "06:00-11:59"
    if 12 <= hh < 18: return "12:00-17:59"
    if 18 <= hh < 24: return "18:00-23:59"
    return "00:00-05:59"


def chi_square_gof(observed_counts, expected_props, label, note=""):
    """Chi-square goodness of fit — with renormalisation for float tolerance."""
    total = sum(observed_counts.values())
    # Renormalise expected proportions to sum to exactly 1
    prop_sum = sum(expected_props.get(k, 0) for k in observed_counts)
    expected = {k: (expected_props.get(k, 0) / prop_sum) * total
                for k in observed_counts}
    obs = np.array([observed_counts[k] for k in observed_counts], dtype=float)
    exp = np.array([expected[k] for k in observed_counts], dtype=float)
    # rescale to eliminate tiny float sum-mismatch
    exp = exp * (obs.sum() / exp.sum())
    mask = exp > 0
    chi2, p = stats.chisquare(obs[mask], exp[mask])
    print(f"\n{label}")
    print(f"  Chi-square = {chi2:.3f}, p = {p:.4f}")
    print(f"  {'Category':<22}{'Observed':>10}{'Expected':>12}{'Diff%':>10}")
    for k in observed_counts:
        diff_pct = 100 * (observed_counts[k] - expected[k]) / expected[k] \
                   if expected[k] > 0 else 0
        print(f"  {k:<22}{observed_counts[k]:>10}{expected[k]:>12.1f}{diff_pct:>9.1f}%")
    if p > 0.05:
        print("  Interpretation: MATCHES anchor (p > 0.05)")
    else:
        print("  Interpretation: DIVERGES from anchor (p <= 0.05)")
    if note:
        print(f"  Note: {note}")
    return p


def main():
    df = pd.read_csv(INPUT_CSV)
    print("=" * 70)
    print("VALIDATION REPORT — Synthetic CMA Accidents 2022-2023")
    print("=" * 70)
    print(f"Total records: {len(df)}")
    print(f"Slave Island records: {(df['Police Division'] == 'Slave Island').sum()}")
    print(f"Divisions covered: {df['Police Division'].nunique()}")

    upshift_note = ("Post-rule severity IS EXPECTED to skew higher than raw "
                    "anchor: rules R1-R3 shift severity upward under joint "
                    "risk conditions. Compare pre-rule (base) severity below.")

    # ---- Test 1: severity marginal (full dataset) ----
    obs = df["Accident Severity"].value_counts().to_dict()
    obs = {k: obs.get(k, 0) for k in ANCHOR_SEVERITY}
    chi_square_gof(obs, ANCHOR_SEVERITY,
                   "TEST 1: Post-rule severity distribution vs Summary Report 2025",
                   note=upshift_note)

    # ---- Test 2: severity — Slave Island slice ----
    si = df[df["Police Division"] == "Slave Island"]
    obs_si = si["Accident Severity"].value_counts().to_dict()
    obs_si = {k: obs_si.get(k, 0) for k in ANCHOR_SEVERITY}
    chi_square_gof(obs_si, ANCHOR_SEVERITY,
                   "TEST 2: Slave Island severity (post-rule) vs Summary Report 2025",
                   note=upshift_note)

    # ---- Test 3: fatal-record time distribution vs Thilakarathna ----
    fatal = df[df["Accident Severity"] == "Fatal"].copy()
    fatal["time_band"] = fatal["Time of Accident"].apply(time_to_band_fatal)
    obs_time = fatal["time_band"].value_counts().to_dict()
    obs_time = {k: obs_time.get(k, 0) for k in ANCHOR_TIME_FATAL}
    chi_square_gof(obs_time, ANCHOR_TIME_FATAL,
                   "TEST 3: Fatal-record time bands vs Thilakarathna 2025")

    # ---- Test 4: joint dependency check — night vs severity ----
    df["hour"] = df["Time of Accident"].apply(lambda t: int(t.split(":")[0]))
    df["is_night"] = df["hour"].apply(lambda h: h >= 18 or h < 6)
    print("\nTEST 4: Joint check — severity by night/day (informational)")
    ct = pd.crosstab(df["is_night"], df["Accident Severity"], normalize="index").round(3)
    print(ct)
    print("  (Rule R1: night+main-road bumps severity; should see slight")
    print("   fatal/serious skew for night rows.)")

    # ---- Test 5: joint check — motorcycle involvement vs severity ----
    df["has_motorcycle"] = df["Vehicle Types"].str.contains("Motorcycle")
    print("\nTEST 5: Joint check — severity by motorcycle involvement")
    ct2 = pd.crosstab(df["has_motorcycle"], df["Accident Severity"],
                      normalize="index").round(3)
    print(ct2)
    print("  (Rule R2: motorcycle involvement bumps severity.)")

    # ---- Test 6: TSTR placeholder ----
    print("\n" + "=" * 70)
    print("TSTR PLACEHOLDER")
    print("=" * 70)
    print("""
To complete TSTR validation:
  1. Load your 50 real Slave Island records into a DataFrame with the
     same column names as this synthetic dataset.
  2. Train a simple classifier (e.g. logistic regression, random forest)
     on the SYNTHETIC Slave Island subset (500 records).
  3. Test it on the 50 REAL records.
  4. Compare accuracy/F1 against the same classifier trained on real
     data via cross-validation.

Example snippet:
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import classification_report

  real = pd.read_csv('real_slave_island_50.csv')
  synth_si = df[df['Police Division'] == 'Slave Island']

  # ... feature-engineer both ...
  clf = RandomForestClassifier(random_state=42).fit(X_synth, y_synth)
  print(classification_report(y_real, clf.predict(X_real)))
""")


if __name__ == "__main__":
    main()