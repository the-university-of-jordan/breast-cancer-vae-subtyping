# -*- coding: utf-8 -*-
"""
Extract the 835-sample malignant subset from Full_Clustered_8744.csv,
using ONLY the unsupervised Cluster_Hier_k7 assignment already produced
by phase4_clustering_8744.py (Full_Clustered run, hierarchical-only) -
never Final_Label. Final_Label is printed afterward purely as a sanity
check, the same way it's used everywhere else in this pipeline: to
interpret a result, never to build it.
"""
import pandas as pd
import os

base_dir = r'C:\Users\user\Desktop\Bioinformatics_Data\8749 samples'

# From the real console output: Cluster_Hier_k7 == 3 is the malignant
# cluster (139 Basal, 66 Her2, 419 LumA, 192 LumB, 18 Normal-like,
# 0 Normal-GTEx, 1 Normal-TCGA = 835 samples).
MALIGNANT_CLUSTER_K7 = 3

# --- 1. Load the existing unsupervised cluster assignments ---
clustered_path = os.path.join(base_dir, 'Clustered_Data_Full_Clustered.csv')
clustered_df = pd.read_csv(clustered_path)

malignant_ids = clustered_df.loc[
    clustered_df['Cluster_Hier_k7'] == MALIGNANT_CLUSTER_K7, 'sampleID'
]
print(f"Samples assigned to Cluster_Hier_k7 == {MALIGNANT_CLUSTER_K7}: {len(malignant_ids)}")

# --- 2. Load the full 5,000-gene matrix and keep only those sample rows ---
full_path = os.path.join(base_dir, 'Full_Clustered_8744.csv')
full_df = pd.read_csv(full_path)

subset_df = full_df[full_df['sampleID'].isin(malignant_ids)].copy()
print(f"Subset shape: {subset_df.shape}  "
      f"(expected: {len(malignant_ids)} rows x 5002 columns)")

# --- 3. Sanity check ONLY - Final_Label was not used to build this subset ---
print("\nFinal_Label breakdown of this subset (confirmation only, "
      "not used to select these rows):")
print(subset_df['Final_Label'].value_counts())

# --- 4. Save ---
out_path = os.path.join(base_dir, 'Malignant_Subset_835.csv')
subset_df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
