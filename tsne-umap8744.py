# -*- coding: utf-8 -*-
"""
Phase 3 - Dimensionality reduction (t-SNE & UMAP) for visualization
Real 8,744-patient primary Analytical Cohort.
Primary: VAE latent space (Z=10,20,40). Secondary: PCA-40 baseline.

Output figures:
  Figure2_UMAP_Comparison.png   - MAIN TEXT Figure 2: one 2x2 panel figure,
                                   UMAP only, PCA_40 vs VAE Z=10/20/40,
                                   PAM50/tissue-label colored, one shared
                                   legend. This is the primary qualitative
                                   validation figure - it lets a reader
                                   compare all four representations against
                                   each other in a single glance.
  FigureS1_tSNE_Comparison.png  - SUPPLEMENTARY: identical 2x2 layout, but
                                   t-SNE (perplexity=40) instead of UMAP -
                                   shows the Figure 2 pattern is not an
                                   artifact of the UMAP algorithm specifically.
  FigureS2_tSNE_Stability.png   - SUPPLEMENTARY: 4 representations x 3
                                   perplexities (30/40/50) in one grid -
                                   shows the pattern is not sensitive to the
                                   perplexity hyperparameter.

Per-representation ER-status figures were removed (decided not to include -
not visually clear); Clinical_Metadata_8744.csv remains available for
Sections 2.6/2.7 (clinicopathological association / survival) independent
of this script.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
import os

# --- 1. SETUP ---
base_dir = r'C:\Users\user\Desktop\Bioinformatics_Data\8749 samples'
files = {
    "PCA_40":  "PCA_Results_8744.csv",
    "VAE_Z10": "VAE_Latent_Z10_8744.csv",
    "VAE_Z20": "VAE_Latent_Z20_8744.csv",
    "VAE_Z40": "VAE_Latent_Z40_8744.csv",
}
PANEL_TITLES = {
    "PCA_40":  "(a) PCA-40",
    "VAE_Z10": "(b) VAE Z=10",
    "VAE_Z20": "(c) VAE Z=20",
    "VAE_Z40": "(d) VAE Z=40",
}
PANEL_ORDER = ["PCA_40", "VAE_Z10", "VAE_Z20", "VAE_Z40"]

PERPLEXITIES = [30, 40, 50]

# Fixed, colorblind-friendly palette (replaces 'husl', which put several
# classes at visually similar hues at this many categories). Assigned once,
# globally, so every figure - main and supplementary - uses identical
# colors for the same class.
ALL_LABELS = ['Normal_GTEX', 'Normal_TCGA', 'Normal', 'LumA', 'LumB', 'Her2', 'Basal']
PALETTE = dict(zip(ALL_LABELS, sns.color_palette("colorblind", len(ALL_LABELS))))

MARKER_SIZE = 10
ALPHA = 0.7

# --- 2. COMPUTE t-SNE (3 perplexities) + UMAP for every representation ---
results = {}  # name -> {'labels', 'sample_ids', 'tsne_runs', 'umap_results'}

for name, filename in files.items():
    print(f"\n--- Processing {name} ---")
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        print(f"Skipping {name}: File not found.")
        continue

    df = pd.read_csv(path)
    X = df.drop(columns=['Final_Label', 'sampleID'], errors='ignore').select_dtypes(include=[np.number])
    labels = df['Final_Label'].reset_index(drop=True)
    sample_ids = df['sampleID'].reset_index(drop=True)

    tsne_runs = {}
    for perp in PERPLEXITIES:
        print(f"Running t-SNE on {name} (perplexity={perp})...")
        tsne = TSNE(
            n_components=2, perplexity=perp, learning_rate='auto',
            max_iter=1000, init='pca', random_state=42
        )
        tsne_runs[perp] = tsne.fit_transform(X)

    print(f"Running UMAP on {name}...")
    reducer = umap.UMAP(
        n_neighbors=30, min_dist=0.3, n_components=2,
        init='random', random_state=42, low_memory=True
    )
    umap_results = reducer.fit_transform(X)

    results[name] = {
        'labels': labels, 'sample_ids': sample_ids,
        'tsne_runs': tsne_runs, 'umap_results': umap_results,
    }

    # --- Save per-representation coordinates (all perplexities + UMAP) ---
    vis_df = pd.DataFrame({'sampleID': sample_ids, 'Final_Label': labels})
    for perp in PERPLEXITIES:
        vis_df[f'tSNE_1_p{perp}'] = tsne_runs[perp][:, 0]
        vis_df[f'tSNE_2_p{perp}'] = tsne_runs[perp][:, 1]
    vis_df['UMAP_1'] = umap_results[:, 0]
    vis_df['UMAP_2'] = umap_results[:, 1]
    vis_df.to_csv(os.path.join(base_dir, f"Vis_Coords_{name}.csv"), index=False)
    print(f"Saved coordinates for {name}.")


# --- 3. PLOTTING HELPERS ---
def scatter_panel(ax, coords, labels, title):
    # Shuffle plotting order so no single class is systematically drawn on
    # top of (or hidden beneath) the others - avoids a misleading impression
    # of separation/overlap caused purely by draw order.
    order = np.random.RandomState(42).permutation(len(labels))
    sns.scatterplot(
        x=coords[order, 0], y=coords[order, 1],
        hue=labels.values[order], palette=PALETTE, hue_order=ALL_LABELS,
        s=MARKER_SIZE, alpha=ALPHA, ax=ax, edgecolor=None,
        legend=False, rasterized=True
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(''); ax.set_ylabel('')
    ax.set_xticks([]); ax.set_yticks([])


def build_shared_legend(fig, ncol=None):
    handles = [
        plt.Line2D([0], [0], marker='o', linestyle='', color=PALETTE[l],
                   markersize=9, label=l)
        for l in ALL_LABELS
    ]
    fig.legend(
        handles=handles, loc='lower center', ncol=ncol or len(ALL_LABELS),
        bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=11
    )


# --- 4. FIGURE 2 (MAIN TEXT): 2x2 UMAP comparison, all four representations ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for ax, name in zip(axes.flat, PANEL_ORDER):
    if name not in results:
        ax.axis('off')
        continue
    r = results[name]
    scatter_panel(ax, r['umap_results'], r['labels'], PANEL_TITLES[name])
build_shared_legend(fig)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(os.path.join(base_dir, "Figure2_UMAP_Comparison.png"), dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved Figure2_UMAP_Comparison.png")

# --- 5. FIGURE S1 (SUPPLEMENTARY): 2x2 t-SNE (perplexity=40) comparison ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for ax, name in zip(axes.flat, PANEL_ORDER):
    if name not in results:
        ax.axis('off')
        continue
    r = results[name]
    scatter_panel(ax, r['tsne_runs'][40], r['labels'], PANEL_TITLES[name])
build_shared_legend(fig)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(os.path.join(base_dir, "FigureS1_tSNE_Comparison.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Saved FigureS1_tSNE_Comparison.png")

# --- 6. FIGURE S2 (SUPPLEMENTARY): perplexity stability, 4 reps x 3 perplexities ---
fig, axes = plt.subplots(4, 3, figsize=(18, 22))
for row, name in enumerate(PANEL_ORDER):
    if name not in results:
        for col in range(3):
            axes[row, col].axis('off')
        continue
    r = results[name]
    for col, perp in enumerate(PERPLEXITIES):
        title = f"{PANEL_TITLES[name]} - perplexity={perp}"
        scatter_panel(axes[row, col], r['tsne_runs'][perp], r['labels'], title)
build_shared_legend(fig)
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(os.path.join(base_dir, "FigureS2_tSNE_Stability.png"), dpi=200, bbox_inches='tight')
plt.close()
print("Saved FigureS2_tSNE_Stability.png")

print("\nAll visualizations generated successfully!")
