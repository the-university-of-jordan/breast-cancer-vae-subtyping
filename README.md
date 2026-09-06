# Transcriptome-Based Subtype Discovery in Breast Cancer Using Variational Representation Learning

Code and derived data supporting the manuscript submitted to *BioMedInformatics* (MDPI). This repository contains the full analysis pipeline used to train variational autoencoder (VAE) representations of bulk breast cancer transcriptomes, benchmark them against a PCA baseline and PAM50 subtype labels, and characterize the resulting clusters clinically, immunologically, and functionally.

## Data source

All expression and clinical data originate from the [UCSC Xena functional genomics explorer](https://xenabrowser.net/) (TCGA breast cancer cohort plus GTEx normal tissue). No new sequencing data were generated. Raw Xena downloads are not redistributed here; the extraction scripts below regenerate the exact processed cohort from the public source.

## Environment

**Python** (VAE training, clustering, dimensionality reduction, survival, clinical association): `pandas`, `numpy`, `scikit-learn`, `scipy`, `torch`, `matplotlib`, `seaborn`, `umap-learn`, `lifelines`.

**R** (differential expression, enrichment, immune deconvolution, gene-panel sensitivity): `data.table`, `matrixStats`, `clusterProfiler`, `org.Hs.eg.db`, `msigdbr`, `fgsea`, `dplyr`, `immunedeconv`, `writexl`, `ggplot2`.

A fixed random seed (42) is used throughout for the train/validation split, t-SNE, UMAP, k-means, GMM, bootstrap resampling, and GSEA, for reproducibility.

## Pipeline order

The scripts are meant to be run in this sequence; each stage's outputs are consumed by the next.

1. **`extract_8744data.py`** — Loads the merged 19,131-sample Xena export, applies the tissue/subtype labeling rule (GTEx → `Normal_GTEX`, TCGA solid normal → `Normal_TCGA`, PAM50 call → subtype, else excluded), and produces the 8,744-patient primary Analytical Cohort (`X_norm_8749_real_patients.csv`).

2. **VAE training, primary cohort** (`vae training8744.py`) — Trains the VAE at Z = 10, 20, 40 on the 8,744-patient cohort (stratified 80/20 train/validation split, seed 42) and computes the PCA-40 baseline. Outputs: `VAE_Latent_Z10_8744.csv`, `VAE_Latent_Z20_8744.csv`, `VAE_Latent_Z40_8744.csv`, `PCA_Results_8744.csv`.

3. **`tsneumap8744.py`** — t-SNE and UMAP projections of all four primary-cohort representations for visualization only (not used for cluster assignment). Outputs: `Figure2_UMAP_Comparison.png`, `FigureS1_tSNE_Comparison.png`, `FigureS2_tSNE_Stability.png`, `Vis_Coords_*.csv`.

4. **`clustering8744.py`** — Hierarchical (Ward), k-means, and GMM clustering (k = 3–8) on each primary-cohort representation, computed on the numeric feature matrix only (labels excluded from clustering input, reattached only for post hoc validation). Outputs: `Clustered_Data_{PCA_40,VAE_Z10,VAE_Z20,VAE_Z40}.csv`, `Clustering_Validation_Results.csv`, `Clustering_Biological_Crosstabs.csv`, `Dendrogram_*.png`.

5. **`extract835data.py`** — Extracts the dominant malignant cluster (`Cluster_Hier_k7 == 3`, determined purely unsupervised in step 4) as the 835-sample malignant-only subset. Output: `Malignant_Subset_835.csv`.

6. **VAE training, malignant subset** (`vae training835.py`) — Retrains the VAE at Z = 10, 20, 40 specifically on the 835-sample subset (adjusted regularization for the smaller sample size). Outputs: `VAE_Latent_Z{10,20,40}_Malignant835.csv`, `PCA_Results_Malignant835.csv`.

7. **`clustering835.py`** — Same three-algorithm clustering procedure as step 4, applied to the malignant-subset-specific representations, displayed at the biologically motivated k = 5. Outputs: `Clustered_Data_*_Malignant.csv`, `Clustering_Validation_Results_MalignantReps.csv`, `Clustering_Biological_Crosstabs_MalignantReps.csv`, `Dendrogram_*.png`.

8. **`concordancetopam835.py`** — Computes ARI/NMI between the malignant-subset Hierarchical k = 5 clusters and PAM50 labels, across k = 3–8. Output: `PAM50_Concordance_MalignantReps.csv`.

9. **`clinical_association835.py`** — Chi-square tests (ER/PR/HER2 status, AJCC stage) and Kruskal-Wallis (age) across the five malignant-subset clusters. Output: `Clinical_Association_Results_MalignantReps.csv`.

10. **`survival_analysis835.py`** — Kaplan-Meier, log-rank test, and Cox proportional hazards (unadjusted and Age+Stage-adjusted, with a likelihood-ratio test for independence) for each representation's clusters. Outputs: `Survival_Analysis_Results_MalignantReps.csv`, `KM_Cluster_VAE_Z10.png`, `KM_Cluster_VAE_Z40.png`, `KM_PAM50.png`.

11. **`Differential_expression835.R`** — limma differential expression, each malignant-subset cluster (VAE Z = 10, Hierarchical k = 5) vs. the rest. Output: `DE_VAE_Z10_Cluster{1-5}_vs_Rest.csv`.

12. **`match_gene_symbols.R`** — Extracts the top 10 driver genes by |t-statistic| per cluster from the DE results. Output: `Top10_DriverGenes_VAE_Z10_AllClusters.xlsx`.

13. **`Enrichment_analysis835.R`** — GO (Biological Process) and KEGG over-representation, plus GSEA against the MSigDB Hallmark collection, per cluster. Outputs: `Cluster{1-5}_mapped_genes.csv`, `Cluster{1-5}_GO_BP.csv`, `Cluster{1-5}_KEGG.csv`, `Cluster{1-5}_GSEA_Hallmark.csv`.

14. **`gesa plot835.R`** — Dot-plot visualization of Hallmark GSEA results across clusters, grouped by biological theme. Output: `Figure_Hallmark_GSEA_Dotplot.png`.

15. **`estimate835.R`** — ESTIMATE stromal/immune/purity scores for the malignant subset.

16. **`quanTIseq_835.R`** — quanTIseq immune cell-type deconvolution (via `immunedeconv`) for the malignant subset. Output: `quanTIseq_Fractions_VAE_Z10_MalignantSubset.csv`.

17. **`genepanel.R`** — Gene-panel-size sensitivity check: rebuilds the malignant-subset expression matrix at 3,000- and 10,000-gene panel sizes (MAD + low-expression filtering). Outputs: `Malignant_Subset_835_3000genes.csv`, `Malignant_Subset_835_10000genes.csv`.

18. **`primary_cohort_stability8744.py`** — Bootstrap robustness check (50 resamples, 85% subsampling) testing whether whole-cohort Hierarchical k = 7 clustering ever recovers PAM50 subtype structure, across all five primary-cohort representations. Outputs: `Step1_PrimaryCohort_BootstrapARI_Summary.csv`, `Step1_PrimaryCohort_BootstrapARI_PerResample.csv`.

## Reproducibility notes

- Cluster assignment was always computed from the numeric representation alone; PAM50/tissue labels were excluded from every clustering input and reattached only after cluster assignment, for validation.
- All stochastic steps (train/validation split, k-means, GMM, t-SNE, UMAP, bootstrap resampling, GSEA) use a fixed seed of 42.
- File paths in the scripts are set to the original local analysis directory and will need to be updated to your local paths (or working directory) to rerun.

## Citation

If you use this code or data, please cite the associated manuscript (citation to be added upon publication).

## License

Code is released under the MIT License. Underlying expression and clinical data remain subject to UCSC Xena's original data use terms.
