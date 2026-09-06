# -*- coding: utf-8 -*-
"""
Phase 4 - Clustering on the real 8,744-patient Analytical Cohort.

Core method: hierarchical clustering (Ward linkage, Euclidean distance -
the only distance Ward's method supports) on each representation's own
feature space: VAE latent Z (10/20/40) and the PCA-40 baseline.

UNSUPERVISED GUARANTEE: clustering is computed from the feature matrix X
alone. Final_Label (PAM50 subtype / tissue origin) is read from the file
but excluded from X and never touches linkage/fcluster/KMeans/GMM - it is
only reattached AFTER cluster assignment, to check biological plausibility.
The console prints exactly which columns were excluded from X on every
run, as an auditable confirmation this held.

Also runs k-means and GMM on the same feature spaces (Section 4.3) so
cluster structure can be checked for method-dependence.

CONSOLE OUTPUT, per representation, in this order:
  1. Internal Validation Metrics table (Hierarchical, k = 3..8):
     columns k, Silhouette, Calinski-Harabasz, Davies-Bouldin.
  2. Biological Subtype Distribution table at k = 7 (Hierarchical).
  3. Biological Subtype Distribution table at k = 7 for k-means.
  4. Biological Subtype Distribution table at k = 7 for GMM.
  5. "Alternative methods on the same feature space: k-means, GMM" table:
     columns k, KMeans Silhouette/CH/DB, GMM Silhouette/CH/DB.

Saved files (unchanged): Clustered_Data_{name}.csv (per-sample cluster
assignments, all three methods, all k), Clustering_Validation_Results.csv
(all metrics, all methods, all k), Clustering_Biological_Crosstabs.csv
(all crosstabs, all methods, all k).
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
    "PCA_40":  "PCA_Results_8744.csv",
    "VAE_Z10": "VAE_Latent_Z10_8744.csv",
    "VAE_Z20": "VAE_Latent_Z20_8744.csv",
    "VAE_Z40": "VAE_Latent_Z40_8744.csv",
}
LABEL_COL = 'Final_Label'   # the only label column carried in these files
ID_COL = 'sampleID'
K_RANGE = range(3, 9)        # k = 3..8, per spec
DISPLAY_K = 7                # the k shown in the biological distribution tables
                              # and used as the dendrogram color-grouping threshold

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

    labels_true = df_raw[LABEL_COL].values  # held out - used only AFTER clustering, never as input
    sample_ids = df_raw[ID_COL].values

    # ============================================================
    # HIERARCHICAL CLUSTERING
    # ============================================================
    Z_linkage = linkage(X, method='ward')

    hier_metric_rows = []
    cluster_assignments = {}
    k7_hier_display = None

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
            k7_hier_display = ct_display

    # --- Table 1: Internal Validation Metrics (Hierarchical) ---
    print_table(f"Table: Internal Validation Metrics (Hierarchical) - {name}",
                pd.DataFrame(hier_metric_rows).round(4))

    # --- Table 2: Biological Subtype Distribution at k=7 (Hierarchical) ---
    print_table(f"Table: Biological Subtype Distribution Across Clusters (k={DISPLAY_K}) - {name} [Hierarchical]",
                k7_hier_display)

    # ============================================================
    # ALTERNATIVE METHODS: k-means, GMM
    # ============================================================
    km_assignments = {}
    gmm_assignments = {}
    km_metric_rows = []
    gmm_metric_rows = []
    k7_km_display = None
    k7_gmm_display = None

    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X)
        gmm = GaussianMixture(n_components=k, random_state=SEED).fit(X)
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
            k7_km_display = ct_km_display
            k7_gmm_display = ct_gmm_display

    # --- Table 3: Biological Subtype Distribution at k=7 (k-means, GMM) ---
    print_table(f"Table: Biological Subtype Distribution Across Clusters (k={DISPLAY_K}) - {name} [KMeans]",
                k7_km_display)
    print_table(f"Table: Biological Subtype Distribution Across Clusters (k={DISPLAY_K}) - {name} [GMM]",
                k7_gmm_display)

    # --- Table 4: Alternative methods on the same feature space: k-means, GMM ---
    alt_table = pd.DataFrame(km_metric_rows).merge(pd.DataFrame(gmm_metric_rows), on='k').round(4)
    print(f"\n--- Alternative methods on the same feature space: k-means, GMM --- ({name})")
    print(alt_table.to_string(index=False))

    # --- SAVE PER-SAMPLE CLUSTER ASSIGNMENTS (all three methods, all k) ---
    out_df = pd.DataFrame({ID_COL: sample_ids, LABEL_COL: labels_true})
    for k in K_RANGE:
        out_df[f'Cluster_Hier_k{k}'] = cluster_assignments[k]
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
metrics_df.to_csv(os.path.join(base_dir, "Clustering_Validation_Results.csv"), index=False)
print("\nSaved: Clustering_Validation_Results.csv")

crosstabs_df = pd.concat(crosstab_rows, ignore_index=True)
crosstabs_df.to_csv(os.path.join(base_dir, "Clustering_Biological_Crosstabs.csv"), index=False)
print("Saved: Clustering_Biological_Crosstabs.csv")

print("\nAll clustering analyses complete.")
