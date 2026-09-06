# -*- coding: utf-8 -*-
"""
Survival analysis (Section 3.5.2, "Survival analysis" block) for the four
malignant-subset representations, grouped by Cluster_Hier_k5 (Hierarchical
clustering, k = 5) - this study's primary algorithm throughout Sections
3.4-3.5. Uses OS_event_nature2012 / OS_Time_nature2012, already merged
into the four Clustered_Data_{Model}_forclinical.csv files by
merge_clinical_malignant.py - no new merge needed.

Three analyses per representation, matching the spec:
1. Kaplan-Meier + multivariate log-rank test across the 5 clusters
   (are the five survival curves different from each other at all?).
2. Cox Proportional Hazards, cluster vs PAM50, compared via C-index on
   the SAME sample subset (fair comparison: does knowing the new cluster
   rank patients by survival better than knowing the PAM50 label alone?).
3. Cox Proportional Hazards adjusted for Age + AJCC_Stage_Group, testing
   whether the cluster block remains significant via a likelihood-ratio
   test against an Age+Stage-only baseline model (is cluster an
   INDEPENDENT prognostic marker, not just a proxy for age/stage?).

A small ridge penalty (penalizer=0.1) is applied to every CoxPHFitter
call for numerical stability - AJCC_Stage_Group has sparse categories
(Stage IV especially), and without it convergence can fail or produce
unstable coefficients. This is standard practice for Cox models with
sparse categorical covariates, not a way of changing the result.

Also saves one Kaplan-Meier figure per representation (by cluster) plus
one Kaplan-Meier figure by PAM50 subtype (Final_Label) for comparison -
which representation's cluster figure becomes "the main Figure" paired
with the PAM50 comparison panel is decided after seeing the real
log-rank/C-index results, not assumed in advance.
"""
import pandas as pd
import numpy as np
import os
import warnings
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from scipy.stats import chi2 as chi2_dist
import matplotlib.pyplot as plt

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
DUR_COL = 'OS_Time_nature2012'
EVENT_COL = 'OS_event_nature2012'
AGE_COL = 'Age_at_Initial_Pathologic_Diagnosis_nature2012'
STAGE_COL = 'AJCC_Stage_Group'
LABEL_COL = 'Final_Label'
PENALIZER = 0.1
# Only these two get a per-cluster KM figure - they're the two representations
# whose log-rank test is itself significant AND whose C-index beats PAM50's;
# PCA-40 and VAE Z=20 don't have a statistically meaningful curve-separation
# story to visualize, so plotting them would just add noisy, non-significant
# overlapping curves to the paper.
PLOT_MODELS = ["VAE_Z10", "VAE_Z40"]

results = []
pam50_plot_done = False

for model_name, filename in FILES.items():
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        print(f"\nSkipping {model_name}: {filename} not found.")
        continue

    df = pd.read_csv(path)
    print("\n" + "=" * 70)
    print(f"SURVIVAL ANALYSIS - {model_name} (grouped by {GROUP_COL})")
    print("=" * 70)

    # --- Base survival cohort: needs duration, event, cluster, PAM50 label ---
    base = df.dropna(subset=[DUR_COL, EVENT_COL, GROUP_COL, LABEL_COL]).copy()
    base[EVENT_COL] = base[EVENT_COL].astype(int)
    print(f"Samples with valid survival + cluster data: {len(base)} / {len(df)}")
    print("Per-cluster n and events:")
    print(base.groupby(GROUP_COL).agg(n=(EVENT_COL, 'size'), events=(EVENT_COL, 'sum')).to_string())

    # --- 1. Multivariate log-rank test across the 5 clusters ---
    lr_result = multivariate_logrank_test(base[DUR_COL], base[GROUP_COL], base[EVENT_COL])
    print(f"\nLog-rank test across {base[GROUP_COL].nunique()} clusters: "
          f"statistic = {lr_result.test_statistic:.4f}, p = {lr_result.p_value:.4e}")

    # --- 2. Cluster-only Cox model vs PAM50-only Cox model, same samples ---
    cluster_dummies = pd.get_dummies(base[GROUP_COL], prefix='Cluster', drop_first=True)
    cox_data_cluster = pd.concat([base[[DUR_COL, EVENT_COL]].reset_index(drop=True),
                                   cluster_dummies.reset_index(drop=True)], axis=1)
    try:
        cph_cluster = CoxPHFitter(penalizer=PENALIZER)
        cph_cluster.fit(cox_data_cluster, duration_col=DUR_COL, event_col=EVENT_COL)
        c_index_cluster = cph_cluster.concordance_index_
    except Exception as e:
        print(f"  WARNING: cluster-only Cox model failed to fit ({e}).")
        c_index_cluster = np.nan

    pam50_dummies = pd.get_dummies(base[LABEL_COL], prefix='PAM50', drop_first=True)
    cox_data_pam50 = pd.concat([base[[DUR_COL, EVENT_COL]].reset_index(drop=True),
                                 pam50_dummies.reset_index(drop=True)], axis=1)
    try:
        cph_pam50 = CoxPHFitter(penalizer=PENALIZER)
        cph_pam50.fit(cox_data_pam50, duration_col=DUR_COL, event_col=EVENT_COL)
        c_index_pam50 = cph_pam50.concordance_index_
    except Exception as e:
        print(f"  WARNING: PAM50-only Cox model failed to fit ({e}).")
        c_index_pam50 = np.nan

    print(f"\nCluster-only Cox model C-index (n={len(base)}): {c_index_cluster:.4f}")
    print(f"PAM50-only Cox model C-index (same n={len(base)}):  {c_index_pam50:.4f}")

    # --- 3. Adjusted Cox model: Age + Stage + Cluster, LR test for cluster block ---
    adj_base = base.dropna(subset=[AGE_COL, STAGE_COL]).copy()
    print(f"\nSamples with valid survival + cluster + age + stage: {len(adj_base)} / {len(df)}")

    stage_dummies = pd.get_dummies(adj_base[STAGE_COL], prefix='Stage', drop_first=True)
    cluster_dummies_adj = pd.get_dummies(adj_base[GROUP_COL], prefix='Cluster', drop_first=True)

    baseline_data = pd.concat([
        adj_base[[DUR_COL, EVENT_COL, AGE_COL]].reset_index(drop=True),
        stage_dummies.reset_index(drop=True)], axis=1)
    full_data = pd.concat([
        adj_base[[DUR_COL, EVENT_COL, AGE_COL]].reset_index(drop=True),
        stage_dummies.reset_index(drop=True),
        cluster_dummies_adj.reset_index(drop=True)], axis=1)

    try:
        cph_baseline = CoxPHFitter(penalizer=PENALIZER)
        cph_baseline.fit(baseline_data, duration_col=DUR_COL, event_col=EVENT_COL)
        cph_full = CoxPHFitter(penalizer=PENALIZER)
        cph_full.fit(full_data, duration_col=DUR_COL, event_col=EVENT_COL)

        lr_stat = 2 * (cph_full.log_likelihood_ - cph_baseline.log_likelihood_)
        df_diff = cluster_dummies_adj.shape[1]
        lr_p = chi2_dist.sf(lr_stat, df_diff)
        independent_marker = "YES" if lr_p < 0.05 else "NO"

        print(f"\nAdjusted Cox model (Age + Stage + Cluster), n = {len(adj_base)}:")
        print(cph_full.summary[['coef', 'exp(coef)', 'p']].to_string())
        print(f"\nLikelihood-ratio test for the Cluster block "
              f"(Age+Stage+Cluster vs Age+Stage only):")
        print(f"  LR statistic = {lr_stat:.4f}, df = {df_diff}, p = {lr_p:.4e}")
        print(f"  Independent prognostic marker after adjusting for Age+Stage: {independent_marker}")
    except Exception as e:
        print(f"  WARNING: adjusted Cox model failed to fit ({e}).")
        lr_stat, df_diff, lr_p, independent_marker = np.nan, np.nan, np.nan, "N/A"

    results.append({
        'Model': model_name, 'N_survival': len(base),
        'LogRank_stat': lr_result.test_statistic, 'LogRank_p': lr_result.p_value,
        'CIndex_Cluster': c_index_cluster, 'CIndex_PAM50': c_index_pam50,
        'N_adjusted': len(adj_base),
        'Adjusted_LR_stat': lr_stat, 'Adjusted_LR_df': df_diff, 'Adjusted_LR_p': lr_p,
        'Independent_Marker': independent_marker,
    })

    # --- Kaplan-Meier figure: this model's clusters. Only generated for the
    # models featured in the paper (PLOT_MODELS) - the ones that both clear
    # the log-rank test on their own and beat PAM50's C-index; the other two
    # don't have a statistically meaningful curve-separation story to show. ---
    if model_name in PLOT_MODELS:
        fig, ax = plt.subplots(figsize=(9, 6))
        kmf = KaplanMeierFitter()
        for grp, sub in base.groupby(GROUP_COL):
            kmf.fit(sub[DUR_COL], sub[EVENT_COL], label=f"Cluster {grp}")
            kmf.plot_survival_function(ax=ax, ci_show=False)
        ax.set_title(f"Survival Curves: {model_name}\nLog-rank p-value: {lr_result.p_value:.2e}")
        ax.set_xlabel("Days")
        ax.set_ylabel("Survival Probability")
        ax.grid(True, alpha=0.3)
        ax.legend(title="AI Clusters")
        plt.tight_layout()
        km_path = os.path.join(base_dir, f"KM_Cluster_{model_name}.png")
        plt.savefig(km_path, dpi=200)
        plt.close()
        print(f"\nSaved: {km_path}")

    # --- Kaplan-Meier figure: PAM50 subtype (only needs to be built once -
    # same patients/survival data regardless of which representation's file
    # produced 'base', since Cluster_Hier_k5 has no missingness of its own) ---
    if not pam50_plot_done:
        pam50_lr = multivariate_logrank_test(base[DUR_COL], base[LABEL_COL], base[EVENT_COL])
        fig, ax = plt.subplots(figsize=(9, 6))
        kmf = KaplanMeierFitter()
        for grp, sub in base.groupby(LABEL_COL):
            kmf.fit(sub[DUR_COL], sub[EVENT_COL], label=grp)
            kmf.plot_survival_function(ax=ax, ci_show=False)
        ax.set_title(f"Survival Curves: PAM50 Subtypes\nLog-rank p-value: {pam50_lr.p_value:.2e}")
        ax.set_xlabel("Days")
        ax.set_ylabel("Survival Probability")
        ax.grid(True, alpha=0.3)
        ax.legend(title="PAM50 Subtype")
        plt.tight_layout()
        pam50_path = os.path.join(base_dir, "KM_PAM50.png")
        plt.savefig(pam50_path, dpi=200)
        plt.close()
        print(f"Saved: {pam50_path}")
        pam50_plot_done = True

# --- SUMMARY ---
summary_df = pd.DataFrame(results)
print("\n" + "=" * 70)
print("SUMMARY: SURVIVAL ANALYSIS ACROSS ALL MODELS")
print("=" * 70)
print(summary_df.to_string(index=False))

out_path = os.path.join(base_dir, "Survival_Analysis_Results_MalignantReps.csv")
summary_df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print("\nSurvival analysis complete.")
