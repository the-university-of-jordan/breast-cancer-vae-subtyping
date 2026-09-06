# -*- coding: utf-8 -*-
"""
Phase 5 (Section 5.1) - Concordance between the malignant-subset clusters
and the PAM50 (Final_Label) subtypes: Adjusted Rand Index (ARI) and
Normalized Mutual Information (NMI), for the four malignant-subset-specific
representations (PCA-40, VAE Z=10/20/40), Hierarchical clustering only -
this study's primary algorithm (Section 2.5).

This does NOT re-run clustering. It reads the per-sample cluster
assignments already saved by phase4_clustering_malignant_reps.py
(Clustered_Data_{name}.csv, one row per sample, columns
Cluster_Hier_k3..k8 alongside Final_Label), so the ARI/NMI numbers are
computed from the exact same cluster assignments already reported in
Sections 3.5/3.5.1 - no new clustering, no risk of a different random
seed producing different clusters.

ARI and NMI are computed across the full k = 3..8 sweep (not just k=5)
so we can see whether PAM50 concordance peaks at the same k already
chosen on biological/mathematical grounds, or somewhere else - this is
reported as an additional cross-check, not a reason to change DISPLAY_K.

Also builds the k=5 contingency table (Cluster x Final_Label) explicitly
via pd.crosstab for each model, to confirm it exactly reproduces the
Table 9 crosstabs already in the manuscript (it must, since it's the same
Cluster_Hier_k5 column).

UNSUPERVISED GUARANTEE: unaffected. Final_Label is only ever compared to
already-computed cluster assignments here, never used to produce them.
"""
import pandas as pd
import numpy as np
import os
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

base_dir = r'C:\Users\user\Desktop\Bioinformatics_Data\8749 samples'

files = {
    "PCA_40_Malignant":  "Clustered_Data_PCA_40_Malignant.csv",
    "VAE_Z10_Malignant": "Clustered_Data_VAE_Z10_Malignant.csv",
    "VAE_Z20_Malignant": "Clustered_Data_VAE_Z20_Malignant.csv",
    "VAE_Z40_Malignant": "Clustered_Data_VAE_Z40_Malignant.csv",
}
LABEL_COL = 'Final_Label'
K_RANGE = range(3, 9)
DISPLAY_K = 5   # same k used throughout Section 3.5 for biological comparison

concordance_rows = []
k5_contingency_tables = {}

for name, filename in files.items():
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        print(f"Skipping {name}: {filename} not found. "
              f"(Re-run phase4_clustering_malignant_reps.py first if this file is missing.)")
        continue

    df = pd.read_csv(path)
    labels_true = df[LABEL_COL].values

    print("\n" + "=" * 60)
    print(f"PAM50 CONCORDANCE FOR: {name} (Hierarchical clustering)")
    print("=" * 60)

    rows = []
    for k in K_RANGE:
        col = f'Cluster_Hier_k{k}'
        if col not in df.columns:
            print(f"  Column {col} not found - skipping k={k}.")
            continue
        cluster_labels = df[col].values
        ari = adjusted_rand_score(labels_true, cluster_labels)
        nmi = normalized_mutual_info_score(labels_true, cluster_labels)
        rows.append({'k': k, 'ARI': ari, 'NMI': nmi})
        concordance_rows.append({'Model': name, 'Method': 'Hierarchical', 'k': k, 'ARI': ari, 'NMI': nmi})

    conc_df = pd.DataFrame(rows).round(4)
    print(f"\nTable: PAM50 Concordance (ARI, NMI) across k, Hierarchical - {name}")
    print(conc_df.to_string(index=False))

    best_row = conc_df.loc[conc_df['ARI'].idxmax()]
    print(f"  -> ARI peaks at k={int(best_row['k'])} (ARI={best_row['ARI']:.4f}); "
          f"value at k={DISPLAY_K}: ARI={conc_df.loc[conc_df.k==DISPLAY_K,'ARI'].values[0]:.4f}, "
          f"NMI={conc_df.loc[conc_df.k==DISPLAY_K,'NMI'].values[0]:.4f}")

    # k=5 contingency table, built directly from the saved cluster column,
    # to confirm it reproduces the Table 9 crosstab already in the manuscript.
    k5_col = f'Cluster_Hier_k{DISPLAY_K}'
    ct = pd.crosstab(df[k5_col], df[LABEL_COL])
    ct.index.name = 'Cluster'
    k5_contingency_tables[name] = ct
    print(f"\nTable: Contingency table (Cluster x {LABEL_COL}) at k={DISPLAY_K} - {name} "
          f"[should match Table 9]")
    print(ct.to_string())

# --- SAVE SUMMARY ---
concordance_df = pd.DataFrame(concordance_rows)
out_path = os.path.join(base_dir, "PAM50_Concordance_MalignantReps.csv")
concordance_df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

print("\n" + "=" * 60)
print("SUMMARY: ARI / NMI at k=5 (Hierarchical), all four representations")
print("=" * 60)
summary = concordance_df[concordance_df.k == DISPLAY_K][['Model', 'ARI', 'NMI']].round(4)
print(summary.to_string(index=False))

print("\nPhase 5 concordance analysis complete.")
