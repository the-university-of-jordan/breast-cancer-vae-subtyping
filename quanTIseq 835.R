## ============================================================
## Phase 7.2 - quanTIseq immune cell deconvolution (immunedeconv)
## VAE_Z10, Hierarchical k=5, 835-sample malignant subset
## ============================================================

## ---- 1. Packages ----
if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
if (!requireNamespace("immunedeconv", quietly = TRUE)) {
  tryCatch({
    remotes::install_github("icbi-lab/immunedeconv")
  }, error = function(e) {
    message("icbi-lab/immunedeconv failed, trying omnideconv/immunedeconv fork...")
    remotes::install_github("omnideconv/immunedeconv")
  })
}
library(immunedeconv)

## ---- 2. Paths ----
base_dir  <- "C:/Users/user/Desktop/Bioinformatics_Data/8749 samples"
expr_file <- file.path(base_dir, "X_norm_READY_FOR_ESTIMATE.csv")
meta_file <- file.path(base_dir, "R_ready_metadata.csv")
out_csv   <- file.path(base_dir, "quanTIseq_Fractions_VAE_Z10_MalignantSubset.csv")

## ---- 3. Read expression matrix (genes as rows, gene symbols in col 1) ----
expr_raw <- read.csv(expr_file, check.names = FALSE, stringsAsFactors = FALSE)
gene_col <- names(expr_raw)[1]
gene_symbols <- expr_raw[[gene_col]]
mat <- as.matrix(expr_raw[, -1, drop = FALSE])
storage.mode(mat) <- "numeric"

if (any(duplicated(gene_symbols))) {
  cat("Collapsing", sum(duplicated(gene_symbols)), "duplicate gene symbol rows by mean expression...\n")
  sum_mat <- rowsum(mat, group = gene_symbols, reorder = FALSE)
  counts  <- table(gene_symbols)[rownames(sum_mat)]
  mat_all <- sum_mat / as.vector(counts)
} else {
  mat_all <- mat
  rownames(mat_all) <- gene_symbols
}
cat("Full expression matrix:", nrow(mat_all), "genes x", ncol(mat_all), "samples\n")

## ---- 4. Read metadata and align to the 835 malignant-subset samples ----
meta <- read.csv(meta_file, check.names = FALSE, stringsAsFactors = FALSE)
sample_col  <- names(meta)[grepl("sample", names(meta), ignore.case = TRUE)][1]
cluster_col <- names(meta)[grepl("cluster", names(meta), ignore.case = TRUE)][1]

matched_cols <- colnames(mat_all) %in% meta[[sample_col]]
mat_835 <- mat_all[, matched_cols, drop = FALSE]

missing_from_expr <- setdiff(meta[[sample_col]], colnames(mat_835))
if (length(missing_from_expr) > 0) {
  stop(length(missing_from_expr), " metadata sample(s) not found in the expression file.")
}

match_idx <- match(colnames(mat_835), meta[[sample_col]])
meta_835  <- meta[match_idx, ]
stopifnot(identical(colnames(mat_835), meta_835[[sample_col]]))
cat("Aligned matrix:", nrow(mat_835), "genes x", ncol(mat_835), "samples (835 expected)\n")

## ---- 5. Convert log2(TPM+1) back to linear TPM for quanTIseq ----
expr_linear <- (2^mat_835) - 1
expr_linear[expr_linear < 0] <- 0   # guard against tiny negative rounding artifacts

## ---- 6. Run quanTIseq ----
quantiseq_res <- deconvolute(expr_linear, method = "quantiseq")
cat("quanTIseq complete. Cell types returned:\n")
print(quantiseq_res$cell_type)

## ---- 7. Reshape to samples-as-rows, reattach cluster label BY POSITION
##         (column order is untouched by deconvolute(), no text round-trip involved,
##         but we stay consistent with the safer approach used in 7.1) ----
cell_fracs <- as.data.frame(t(quantiseq_res[, -1, drop = FALSE]))
colnames(cell_fracs) <- quantiseq_res$cell_type

stopifnot(nrow(cell_fracs) == nrow(meta_835))
cell_fracs[[sample_col]]  <- meta_835[[sample_col]]
cell_fracs[[cluster_col]] <- meta_835[[cluster_col]]

final_df <- cell_fracs[, c(sample_col, setdiff(names(cell_fracs), c(sample_col, cluster_col)), cluster_col)]

write.csv(final_df, out_csv, row.names = FALSE)
cat("\nSaved:", out_csv, "\n")
cat("Preview:\n")
print(head(final_df))