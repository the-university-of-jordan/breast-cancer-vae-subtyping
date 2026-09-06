# -*- coding: utf-8 -*-
"""
Phase 4 (Section 3.5) - Clustering the malignant-only subset's OWN
PCA-40 and VAE (Z=10/20/40) representations, refit specifically on the
835 malignant samples (see train_vae_malignant835.py) - NOT reused or
row-subset from the whole-cohort models.

Same core method as phase4_clustering_8744.py: hierarchical clustering
(Ward linkage) on each representation's own feature space, cross-checked
with k-means and GMM (RUN_ALTERNATIVE_METHODS = True here, since the
whole point of this run is the same three-algorithm cross-validation
used for the raw-gene malignant-subset result in Section 3.4).

UNSUPERVISED GUARANTEE: unchanged - Final_Label is dropped from the
feature matrix X and never touches linkage/fcluster/KMeans/GMM; it is
reattached only after cluster assignment, to build the biological
crosstabs. The excluded columns are printed on every run.

DISPLAY_K = 5, not 7: this matches the biologically-motivated k already
established for the malignant subset in Section 3.4 (raw 5,000-gene
result), so these new representations are compared to that result at
the same k, apples-to-apples. All of k = 3..8 is still computed and
saved for every representation/method, in case a different k turns out
to be better mathematically or biologically for one of these
representations specifically.

Saved files (same pattern as before): Clustered_Data_{name}.csv,
Clustering_Validation_Results.csv, Clustering_Biological_Crosstabs.csv,
Dendrogram_{name}.png.
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

SEED = 42
np.random.seed(SEED)

# --- 1. SETUP ---
base_dir = r'C:\Users\user\Desktop\Bioinformatics_Data\8749 samples'
files = {
    "PCA_40_Malignant":  "PCA_Results_Malignant835.csv",
    "VAE_Z10_Malignant": "VAE_Latent_Z10_Malignant835.csv",
    "VAE_Z20_Malignant": "VAE_Latent_Z20_Malignant835.csv",
    "VAE_Z40_Malignant": "VAE_Latent_Z40_Malignant835.csv",
    # Malignant_Subset (raw 5,000-gene) already clustered and analyzed in
    # Section 3.4 - left out here so this run only processes the four new
    # malignant-subset-specific representations. Uncomment to re-run it too.
    # "Malignant_Subset": "Malignant_Subset_835.csv",
}
LABEL_COL = 'Final_Label'   # the only label column carried in these files
ID_COL = 'sampleID'
K_RANGE = range(3, 9)        # k = 3..8, per spec
DISPLAY_K = 5                # matches the biologically-chosen k from Section 3.4
                              # (raw-gene malignant subset), for apples-to-apples
                              # comparison across representations

# Set to True to also run the k-means/GMM robustness check. True here,
# since cross-algorithm validation at k=5 is central to this comparison
# (same reasoning as Section 3.4).
RUN_ALTERNATIVE_METHODS = True

# GaussianMixture defaults to covariance_type='full'. All four files here
# are 10-40 dimensional (PCA-40, VAE Z=10/20/40), so full covariance is
# fine - no diagonal-covariance override needed (unlike the raw 5,000-gene
# Malignant_Subset run).
GMM_COV_TYPE = {}

LABEL_COLORS = {
    'Normal_GTEX': '#0173B2', 'Normal_TCGA': '#DE8F05', 'Normal': '#029E73',
    'LumA': '#D55E00', 'LumB': '#CC78BC', 'Her2': '#CA9161', 'Basal': '#FBAFE4',
}

metrics_rows = []
crosstab_rows = []


def print_table(title, df):
    print(f"\n{title}")
    print(df.to_string(index=False))


def biological_crosstab(clusters, labels_true, model_name, method_name, k):
    """Build the Cluster x Final_Label table. Returns (display_df, store_df) -
    display_df is clean for console; store_df carries Model/Method/k for the
    combined CSV export."""
    ct = pd.crosstab(pd.Series(clusters, name='Cluster'), pd.Series(labels_true, name=LABEL_COL))
    ct_store = ct.copy()
    ct_store.insert(0, 'Model', model_name)
    ct_store.insert(1, 'Method', method_name)
    ct_store.insert(2, 'k', k)
    return ct.reset_index(), ct_store.reset_index()


for name, filename in files.items():
    print("\n" + "=" * 60)
    print(f"ANALYSIS FOR: {name}")
    print("=" * 60)

    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        print(f"Skipping {name}: File not found.")
        continue

    df_raw = pd.read_csv(path)

    # --- UNSUPERVISED FEATURE MATRIX ---
    X_df = df_raw.drop(columns=[LABEL_COL, ID_COL], errors='ignore').select_dtypes(include=[np.number])
    dropped_non_feature = sorted(set(df_raw.columns) - set(X_df.columns))
    print(f"Columns excluded from clustering input (label/ID, not used for distance): {dropped_non_feature}")
    X = X_df.values
    print(f"Clustering feature matrix shape: {X.shape}")
    if X.shape[1] > 500:
        print("Note: high-dimensional feature space - Ward linkage and the "
              "k-means/GMM sweep will take noticeably longer to run here "
              "than for the 10-40 dimensional representations.")

    labels_true = df_raw[LABEL_COL].values  # held out - used only AFTER clustering, never as input
    sample_ids = df_raw[ID_COL].values

    # ============================================================
    # HIERARCHICAL CLUSTERING
    # ============================================================
    Z_linkage = linkage(X, method='ward')

    hier_metric_rows = []
    cluster_assignments = {}
    k5_hier_display = None

    for k in K_RANGE:
        clusters = fcluster(Z_linkage, k, criterion='maxclust')
        cluster_assignments[k] = clusters

        sample_idx = np.random.choice(len(X), size=min(10000, len(X)), replace=False)
        sil = silhouette_score(X[sample_idx], clusters[sample_idx])
        ch_score = calinski_harabasz_score(X, clusters)
        db_score = davies_bouldin_score(X, clusters)

        hier_metric_rows.append({'k': k, 'Silhouette': sil, 'Calinski_Harabasz': ch_score, 'Davies_Bouldin': db_score})
        metrics_rows.append({'Model': name, 'Method': 'Hierarchical', 'k': k,
                              'Silhouette': sil, 'Calinski_Harabasz': ch_score, 'Davies_Bouldin': db_score})

        ct_display, ct_store = biological_crosstab(clusters, labels_true, name, 'Hierarchical', k)
        crosstab_rows.append(ct_store)
        if k == DISPLAY_K:
            k5_hier_display = ct_display

    # --- Table 1: Internal Validation Metrics (Hierarchical) ---
    print_table(f"Table: Internal Validation Metrics (Hierarchical) - {name}",
                pd.DataFrame(hier_metric_rows).round(4))

    # --- Table 2: Biological Subtype Distribution at k=5 (Hierarchical) ---
    print_table(f"Table: Biological Subtype Distribution Across Clusters (k={DISPLAY_K}) - {name} [Hierarchical]",
                k5_hier_display)

    # ============================================================
    # ALTERNATIVE METHODS: k-means, GMM
    # ============================================================
    km_assignments = {}
    gmm_assignments = {}

    if RUN_ALTERNATIVE_METHODS:
        km_metric_rows = []
        gmm_metric_rows = []
        k5_km_display = None
        k5_gmm_display = None

        cov_type = GMM_COV_TYPE.get(name, 'full')
        print(f"GMM covariance_type for {name}: '{cov_type}'"
              + ("  (diagonal - full covariance is infeasible at this dimensionality)"
                 if cov_type != 'full' else ""))

        for k in K_RANGE:
            km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X)
            gmm = GaussianMixture(n_components=k, covariance_type=cov_type, random_state=SEED).fit(X)
            gmm_labels = gmm.predict(X)

            km_assignments[k] = km.labels_
            gmm_assignments[k] = gmm_labels

            sample_idx = np.random.choice(len(X), size=min(10000, len(X)), replace=False)
            km_sil = silhouette_score(X[sample_idx], km.labels_[sample_idx])
            gmm_sil = silhouette_score(X[sample_idx], gmm_labels[sample_idx])
            km_ch, km_db = calinski_harabasz_score(X, km.labels_), davies_bouldin_score(X, km.labels_)
            gmm_ch, gmm_db = calinski_harabasz_score(X, gmm_labels), davies_bouldin_score(X, gmm_labels)

            km_metric_rows.append({'k': k, 'KMeans_Silhouette': km_sil, 'KMeans_CH': km_ch, 'KMeans_DB': km_db})
            gmm_metric_rows.append({'k': k, 'GMM_Silhouette': gmm_sil, 'GMM_CH': gmm_ch, 'GMM_DB': gmm_db})

            metrics_rows.append({'Model': name, 'Method': 'KMeans', 'k': k,
                                  'Silhouette': km_sil, 'Calinski_Harabasz': km_ch, 'Davies_Bouldin': km_db})
            metrics_rows.append({'Model': name, 'Method': 'GMM', 'k': k,
                                  'Silhouette': gmm_sil, 'Calinski_Harabasz': gmm_ch, 'Davies_Bouldin': gmm_db})

            ct_km_display, ct_km_store = biological_crosstab(km.labels_, labels_true, name, 'KMeans', k)
            ct_gmm_display, ct_gmm_store = biological_crosstab(gmm_labels, labels_true, name, 'GMM', k)
            crosstab_rows.append(ct_km_store)
            crosstab_rows.append(ct_gmm_store)
            if k == DISPLAY_K:
                k5_km_display = ct_km_display
                k5_gmm_display = ct_gmm_display

        # --- Table 3: Biological Subtype Distribution at k=5 (k-means, GMM) ---
        print_table(f"Table: Biological Subtype Distribution Across Clusters (k={DISPLAY_K}) - {name} [KMeans]",
                    k5_km_display)
        print_table(f"Table: Biological Subtype Distribution Across Clusters (k={DISPLAY_K}) - {name} [GMM]",
                    k5_gmm_display)

        # --- Table 4: Alternative methods on the same feature space: k-means, GMM ---
        alt_table = pd.DataFrame(km_metric_rows).merge(pd.DataFrame(gmm_metric_rows), on='k').round(4)
        print(f"\n--- Alternative methods on the same feature space: k-means, GMM --- ({name})")
        print(alt_table.to_string(index=False))
    else:
        print("\n(k-means/GMM skipped this run - RUN_ALTERNATIVE_METHODS = False)")

    # --- SAVE PER-SAMPLE CLUSTER ASSIGNMENTS ---
    out_df = pd.DataFrame({ID_COL: sample_ids, LABEL_COL: labels_true})
    for k in K_RANGE:
        out_df[f'Cluster_Hier_k{k}'] = cluster_assignments[k]
        if RUN_ALTERNATIVE_METHODS:
            out_df[f'Cluster_KMeans_k{k}'] = km_assignments[k]
            out_df[f'Cluster_GMM_k{k}'] = gmm_assignments[k]
    out_path = os.path.join(base_dir, f"Clustered_Data_{name}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # ============================================================
    # DENDROGRAM WITH Final_Label COLOR BAR
    # ============================================================
    print("Building dendrogram figure...")
    dendro = dendrogram(Z_linkage, no_plot=True)
    leaf_order = dendro['leaves']

    fig, (ax_dendro, ax_bar) = plt.subplots(
        2, 1, figsize=(16, 8), gridspec_kw={'height_ratios': [5, 0.4]}
    )
    threshold = Z_linkage[-(DISPLAY_K - 1), 2] if len(Z_linkage) >= DISPLAY_K - 1 else None
    dendrogram(Z_linkage, ax=ax_dendro, no_labels=True, color_threshold=threshold)
    ax_dendro.set_title(f'Hierarchical Clustering Dendrogram - {name} (Ward linkage)')
    ax_dendro.set_ylabel('Ward distance')
    ax_dendro.set_xticks([])

    bar_colors = np.array([mcolors.to_rgb(LABEL_COLORS.get(labels_true[i], '#CCCCCC')) for i in leaf_order])
    ax_bar.imshow(bar_colors[np.newaxis, :, :], aspect='auto', extent=[0, len(leaf_order), 0, 1])
    ax_bar.set_yticks([]); ax_bar.set_xticks([])
    ax_bar.set_xlabel('Samples (dendrogram leaf order)')

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in LABEL_COLORS.values()]
    fig.legend(handles, LABEL_COLORS.keys(), loc='upper right', ncol=1, fontsize=8, bbox_to_anchor=(1.12, 0.95))

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f"Dendrogram_{name}.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: Dendrogram_{name}.png")

# --- SAVE SUMMARY METRICS + CROSSTABS ---
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(os.path.join(base_dir, "Clustering_Validation_Results_MalignantReps.csv"), index=False)
print("\nSaved: Clustering_Validation_Results_MalignantReps.csv")

crosstabs_df = pd.concat(crosstab_rows, ignore_index=True)
crosstabs_df.to_csv(os.path.join(base_dir, "Clustering_Biological_Crosstabs_MalignantReps.csv"), index=False)
print("Saved: Clustering_Biological_Crosstabs_MalignantReps.csv")

print("\nAll clustering analyses complete.")
