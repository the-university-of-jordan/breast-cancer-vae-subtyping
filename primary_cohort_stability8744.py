# -*- coding: utf-8 -*-
"""
Step 1 (Phase 8, primary-cohort robustness check): bootstrap-resample the
whole 8,744-sample primary cohort and test whether Hierarchical (Ward) k=7
clustering EVER reliably recovers PAM50 SUBTYPE structure, across all five
representations already used in Sections 3.3/3.4 (raw 5000-gene, PCA-40,
VAE Z=10/20/40).

Design (Option 1, agreed with the user): Section 3.3/3.4 already show, on
the single full-sample run, that no representation fragments the
malignant/PAM50-associated population into subtype-resolved sub-clusters at
k=7. This script tests whether that NULL result is itself stable across 50
independent 85%-subsamples, rather than a fluke of one particular run. For
every resample, we recompute Hierarchical Ward k=7 clustering on the WHOLE
resampled cohort (exactly as done in the original analysis - malignant and
non-malignant samples together), then compute ARI/NMI between the resulting
cluster labels and PAM50 SUBTYPE identity, restricted to samples with a true
PAM50 subtype call (Basal, Her2, LumA, LumB, Normal) - excluding Normal_TCGA
and Normal_GTEx, which are tissue-origin labels, not subtype calls.

If ARI/NMI stay consistently near zero across all 50 resamples for every
representation, that is reproducible evidence that whole-cohort clustering
cannot recover subtype structure regardless of subsample or representation -
directly supporting the pivot to the malignant-only subset.
"""
import pandas as pd
import numpy as np
import os
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

base_dir = r'C:\Users\user\Desktop\Bioinformatics_Data\8749 samples\step1 robustness'
K = 7
N_BOOTSTRAP = 50
SUBSAMPLE_FRAC = 0.85
SEED = 42

# True PAM50 molecular subtype calls (NOT tissue-origin labels). Confirmed
# against the actual Final_Label vocabulary in the 8744-cohort files: 7
# total values - BASAL, HER2, LUM A, LUM B, NORMAL, NORMAL GTEX, NORMAL TCGA.
# NORMAL GTEX and NORMAL TCGA are tissue-origin labels, not PAM50 subtype
# calls, so they are excluded here.
PAM50_SUBTYPES = ['Basal', 'Her2', 'LumA', 'LumB', 'Normal']
REPRESENTATIONS = [
    ('Full_Clustered_8744.csv',       'raw_5000genes', 'ENSG'),   # feature cols start with ENSG
    ('Clustered_Data_PCA_40.csv',     'PCA_40',        'PC'),     # feature cols start with PC
    ('Clustered_Data_VAE_Z10.csv',    'VAE_Z10',       'Z'),      # feature cols start with Z
    ('Clustered_Data_VAE_Z20.csv',    'VAE_Z20',       'Z'),
    ('Clustered_Data_VAE_Z40.csv',    'VAE_Z40',       'Z'),
]

known_non_feature_cols = ['sampleID', 'Final_Label', 'Cluster_Hier_k7']

rng = np.random.default_rng(SEED)
all_resample_rows = []
summary_rows = []

for file_name, tag, prefix in REPRESENTATIONS:
    path = os.path.join(base_dir, file_name)
    if not os.path.exists(path):
        print(f"\u26a0 Skipping {tag} - file not found: {path}")
        continue

    print(f"\n{'='*70}\n{tag}  ({file_name})\n{'='*70}")
    df = pd.read_csv(path)
    print("Final_Label value counts:")
    print(df['Final_Label'].value_counts())

    feature_cols = [c for c in df.columns
                    if c not in known_non_feature_cols and c.startswith(prefix)]
    print(f"Detected {len(feature_cols)} feature columns (prefix '{prefix}').")

    X_full = df[feature_cols].apply(pd.to_numeric, errors='coerce').values
    is_subtype = df['Final_Label'].isin(PAM50_SUBTYPES).values
    print(f"n = {len(df)} total | {is_subtype.sum()} with a true PAM50 subtype call.")

    # --- Sanity check on the FULL sample using the already-stored reference
    #     Cluster_Hier_k7 assignment, before running any bootstrap ---
    if 'Cluster_Hier_k7' in df.columns:
        ref_ari = adjusted_rand_score(df['Final_Label'][is_subtype], df['Cluster_Hier_k7'][is_subtype])
        ref_nmi = normalized_mutual_info_score(df['Final_Label'][is_subtype], df['Cluster_Hier_k7'][is_subtype])
        print(f"Reference (full-sample, stored Cluster_Hier_k7) ARI = {ref_ari:.4f} | NMI = {ref_nmi:.4f}")
        print("(Expect this near zero, matching Section 3.3/3.4's qualitative finding "
              "that no configuration fragments the malignant cluster by PAM50 subtype at k=7.)")

    # --- Bootstrap loop ---
    n_total = len(df)
    n_sub = int(round(SUBSAMPLE_FRAC * n_total))
    aris, nmis = [], []

    for b in range(N_BOOTSTRAP):
        idx = rng.choice(n_total, size=n_sub, replace=False)
        X_sub = X_full[idx]
        label_sub = df['Final_Label'].values[idx]
        subtype_mask_sub = is_subtype[idx]

        Z_link = linkage(X_sub, method='ward')
        clusters_sub = fcluster(Z_link, t=K, criterion='maxclust')

        if subtype_mask_sub.sum() > 1:
            ari = adjusted_rand_score(label_sub[subtype_mask_sub], clusters_sub[subtype_mask_sub])
            nmi = normalized_mutual_info_score(label_sub[subtype_mask_sub], clusters_sub[subtype_mask_sub])
        else:
            ari, nmi = np.nan, np.nan
        aris.append(ari)
        nmis.append(nmi)

        if (b + 1) % 10 == 0:
            print(f"  resample {b+1}/{N_BOOTSTRAP} done "
                  f"(running mean ARI so far: {np.nanmean(aris):.4f})")

        all_resample_rows.append({'Representation': tag, 'Resample': b + 1,
                                   'n_subsample': n_sub, 'ARI': ari, 'NMI': nmi})

    aris = np.array(aris, dtype=float)
    nmis = np.array(nmis, dtype=float)
    print(f"\n{tag} summary over {N_BOOTSTRAP} resamples:")
    print(f"  ARI: mean={np.nanmean(aris):.4f}  sd={np.nanstd(aris):.4f}  "
          f"min={np.nanmin(aris):.4f}  max={np.nanmax(aris):.4f}")
    print(f"  NMI: mean={np.nanmean(nmis):.4f}  sd={np.nanstd(nmis):.4f}  "
          f"min={np.nanmin(nmis):.4f}  max={np.nanmax(nmis):.4f}")

    summary_rows.append({
        'Representation': tag, 'n_full': n_total, 'n_subsample': n_sub,
        'ARI_mean': round(np.nanmean(aris), 4), 'ARI_sd': round(np.nanstd(aris), 4),
        'ARI_min': round(np.nanmin(aris), 4), 'ARI_max': round(np.nanmax(aris), 4),
        'NMI_mean': round(np.nanmean(nmis), 4), 'NMI_sd': round(np.nanstd(nmis), 4),
        'NMI_min': round(np.nanmin(nmis), 4), 'NMI_max': round(np.nanmax(nmis), 4),
    })

summary_df = pd.DataFrame(summary_rows)
print(f"\n{'='*70}\nSTEP 1 SUMMARY (Table 20 material)\n{'='*70}")
print(summary_df.to_string(index=False))

summary_df.to_csv(os.path.join(base_dir, "Step1_PrimaryCohort_BootstrapARI_Summary.csv"), index=False)
pd.DataFrame(all_resample_rows).to_csv(os.path.join(base_dir, "Step1_PrimaryCohort_BootstrapARI_PerResample.csv"), index=False)
print("\nSaved: Step1_PrimaryCohort_BootstrapARI_Summary.csv")
print("Saved: Step1_PrimaryCohort_BootstrapARI_PerResample.csv")
