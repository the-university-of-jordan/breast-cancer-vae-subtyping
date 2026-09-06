# -*- coding: utf-8 -*-
"""
Clinical association analysis (Section 3.5.2) for the four malignant-subset
representations, grouped by Cluster_Hier_k5 (Hierarchical clustering,
k = 5) - this study's primary algorithm throughout Sections 3.4-3.5.
KMeans/GMM cluster columns are not used here (these files don't carry
them anyway).

Categorical features (ER/PR/HER2 status, AJCC stage group): tested with
Chi-square throughout, per your instruction to use Chi-square rather
than a Fisher's-exact-style fallback. Note this is a deliberate
simplification: because the group variable always has 5 levels, every
contingency table here is at least 5xC, so scipy's fisher_exact (2x2
only) could never have been used as a small-cell fallback anyway - a
generalized Fisher-Irwin exact test for RxC tables isn't implemented in
scipy, and no R environment is available in this pipeline. The fraction
of expected cells below 5 (the standard adequacy rule for the
chi-square approximation) is still computed and printed for every
feature/model as a diagnostic - not to switch tests, just so any test
run on a sparse table (this will happen for AJCC Stage, where Stage IV
is rare) is flagged rather than silently reported as if the assumption
held.

ER/PR/HER2 status are binarized to Positive/Negative for this test
(Indeterminate/Equivocal treated as missing for that feature only,
standard practice in breast-cancer clinical association analyses).
AJCC_Stage_Group (I/II/III/IV) is used as already built by
merge_clinical_malignant.py.

Age (continuous) is tested with Kruskal-Wallis across the five clusters,
no fallback needed - it is already the nonparametric, small-sample-safe
choice.

No Grade column exists in this clinical file (confirmed in Step 1) and
is not tested.
"""
import pandas as pd
import numpy as np
import os
from scipy.stats import chi2_contingency, kruskal

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

base_dir = r'C:\Users\user\Desktop\Bioinformatics_Data\8749 samples'
FILES = {
    "PCA_40":  "Clustered_Data_PCA_40_forclinical.csv",
    "VAE_Z10": "Clustered_Data_VAE_Z10_forclinical.csv",
    "VAE_Z20": "Clustered_Data_VAE_Z20_forclinical.csv",
    "VAE_Z40": "Clustered_Data_VAE_Z40_forclinical.csv",
}

GROUP_COL = 'Cluster_Hier_k5'
AGE_COL = 'Age_at_Initial_Pathologic_Diagnosis_nature2012'
EXPECTED_CELL_MIN = 5
SPARSE_CELL_FRACTION_THRESHOLD = 0.20  # >20% of cells below EXPECTED_CELL_MIN -> flagged as a caveat


def binarize_status(series):
    """Positive/Negative only; everything else (Indeterminate, Equivocal, NaN) -> NaN."""
    return series.where(series.isin(['Positive', 'Negative']))


def test_categorical(df, group_col, feat_col, label):
    temp = df.dropna(subset=[group_col, feat_col])
    n_used = len(temp)
    table = pd.crosstab(temp[group_col], temp[feat_col])

    if table.shape[0] < 2 or table.shape[1] < 2:
        return {'Feature': label, 'n': n_used, 'Test': 'N/A (degenerate table)',
                'Statistic': np.nan, 'p_value': np.nan, 'Table': table}

    stat, p_chi2, dof, expected = chi2_contingency(table)
    sparse_fraction = (expected < EXPECTED_CELL_MIN).sum() / expected.size
    flag = f" [CAVEAT: {sparse_fraction:.0%} of expected cells <5]" if sparse_fraction > SPARSE_CELL_FRACTION_THRESHOLD else ""

    return {'Feature': label, 'n': n_used,
            'Test': f'Chi-square{flag}',
            'Statistic': stat, 'p_value': p_chi2, 'Table': table}


def test_age(df, group_col, age_col):
    temp = df.dropna(subset=[group_col, age_col])
    n_used = len(temp)
    groups = [g[age_col].values for _, g in temp.groupby(group_col)]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return {'Feature': 'Age', 'n': n_used, 'Test': 'N/A (fewer than 2 non-empty groups)',
                'Statistic': np.nan, 'p_value': np.nan}
    stat, p = kruskal(*groups)
    medians = temp.groupby(group_col)[age_col].median()
    return {'Feature': 'Age', 'n': n_used, 'Test': 'Kruskal-Wallis',
            'Statistic': stat, 'p_value': p, 'Medians_by_cluster': medians}


all_results = []

for model_name, filename in FILES.items():
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        print(f"\nSkipping {model_name}: {filename} not found.")
        continue

    df = pd.read_csv(path)
    if GROUP_COL not in df.columns:
        print(f"\nSkipping {model_name}: '{GROUP_COL}' column not found.")
        continue

    df = df.copy()
    df['ER_Binary'] = binarize_status(df['ER_Status_nature2012'])
    df['PR_Binary'] = binarize_status(df['PR_Status_nature2012'])
    df['HER2_Binary'] = binarize_status(df['HER2_Final_Status_nature2012'])

    print("\n" + "=" * 70)
    print(f"CLINICAL ASSOCIATION - {model_name} (grouped by {GROUP_COL})")
    print("=" * 70)

    cat_features = [('ER_Binary', 'ER Status'), ('PR_Binary', 'PR Status'),
                     ('HER2_Binary', 'HER2 Status'), ('AJCC_Stage_Group', 'AJCC Stage')]

    for col, label in cat_features:
        result = test_categorical(df, GROUP_COL, col, label)
        result['Model'] = model_name
        all_results.append(result)

        print(f"\n--- {label} ---")
        print(f"n used: {result['n']}   Test: {result['Test']}")
        if not np.isnan(result['p_value']):
            print(f"Statistic: {result['Statistic']:.4f}   p-value: {result['p_value']:.4e}")
        if 'Table' in result:
            print("Contingency table (rows = cluster, columns = category):")
            print(result['Table'].to_string())

    age_result = test_age(df, GROUP_COL, AGE_COL)
    age_result['Model'] = model_name
    all_results.append(age_result)
    print(f"\n--- Age ---")
    print(f"n used: {age_result['n']}   Test: {age_result['Test']}")
    if not np.isnan(age_result['p_value']):
        print(f"Statistic: {age_result['Statistic']:.4f}   p-value: {age_result['p_value']:.4e}")
    if 'Medians_by_cluster' in age_result:
        print("Median age by cluster:")
        print(age_result['Medians_by_cluster'].to_string())

# --- SUMMARY TABLE ---
summary_rows = []
for r in all_results:
    summary_rows.append({
        'Model': r['Model'], 'Feature': r['Feature'], 'n': r['n'],
        'Test': r['Test'], 'Statistic': r['Statistic'], 'p_value': r['p_value'],
    })
summary_df = pd.DataFrame(summary_rows)

print("\n" + "=" * 70)
print("SUMMARY: CLINICAL ASSOCIATION p-VALUES (Cluster_Hier_k5)")
print("=" * 70)
print(summary_df.to_string(index=False))

out_path = os.path.join(base_dir, "Clinical_Association_Results_MalignantReps.csv")
summary_df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print("\nClinical association analysis complete.")
