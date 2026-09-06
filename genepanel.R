# -*- coding: utf-8 -*-
# Step 4 (Phase 8.2, gene-panel-size sensitivity) - v3: memory-safe version.
# The file itself fits in RAM (loading beforefilter.csv as a data.table worked
# fine); the crash came from ALSO building a full 60,498 x 19,131 numeric
# matrix copy of it (as.matrix() on everything at once), which needs its own
# ~8.6 GB block on top of the table already in memory. Fix: compute the
# per-gene low-expression / MAD statistics in row-batches (a few thousand
# genes at a time), and only ever materialize a full matrix for the small
# number of genes we actually keep (3000 / 10000 rows).

if (!requireNamespace("data.table", quietly = TRUE)) install.packages("data.table")
if (!requireNamespace("matrixStats", quietly = TRUE)) install.packages("matrixStats")
library(data.table)
library(matrixStats)

base_dir <- "C:/Users/user/Desktop/Bioinformatics_Data/8749 samples"
merged_path <- file.path(base_dir, "X_Merged_With_Labels.csv")

# ---- 1. Load beforefilter.csv (this part already worked) ----
cat("Loading beforefilter.csv - this can take a few minutes and several GB of RAM...\n")
dt <- fread(file.path(base_dir, "beforefilter.csv"), header = TRUE)

id_col  <- names(dt)[1]   # "sample" column -> actually holds bare Ensembl gene IDs
sym_col <- names(dt)[2]   # "gene" column   -> gene symbols (kept for reference only)
sample_cols <- names(dt)[3:ncol(dt)]
cat("Detected", length(sample_cols), "sample columns and", nrow(dt), "genes.\n")

# ---- 2. Get the TRUE 19,131-sample whitelist from X_Merged_With_Labels.csv's
#         sampleID COLUMN VALUES ----
cat("Reading sampleID column from X_Merged_With_Labels.csv...\n")
sample_whitelist <- fread(merged_path, select = "sampleID")[["sampleID"]]
cat("X_Merged_With_Labels.csv has", length(sample_whitelist), "sample rows.\n")

ref_samples <- intersect(sample_cols, sample_whitelist)
cat(length(ref_samples), "of", length(sample_cols),
    "beforefilter.csv sample columns matched X_Merged_With_Labels.csv's sampleID values.\n")
if (length(ref_samples) < 0.99 * length(sample_whitelist)) {
  warning("Expected close to ", length(sample_whitelist), " matching samples, found ",
          length(ref_samples), ". Stop and paste this warning back before trusting the rest.")
}
sample_cols <- ref_samples

# ---- 3. Get the EXISTING 5000-gene panel's Ensembl IDs from
#         X_Merged_With_Labels.csv's header (ground truth for the sanity check) ----
merged_header <- names(fread(merged_path, nrows = 0))
is_gene_col <- grepl("^ENSG[0-9]+$", merged_header)
existing_5000_genes <- merged_header[is_gene_col]
clinical_cols <- setdiff(merged_header, c("sampleID", existing_5000_genes))
cat("X_Merged_With_Labels.csv: ", length(existing_5000_genes), " gene columns (ENSG), ",
    length(clinical_cols), " clinical columns: ", paste(clinical_cols, collapse = ", "), "\n", sep = "")

# ---- 4. Low-expression filter + MAD, computed in ROW BATCHES (memory-safe) ----
# beforefilter.csv is log2(TPM + 0.001). "TPM < 1" on that scale means
# log2(TPM + 0.001) < log2(1.001).
CUTOFF <- log2(1 + 0.001)
cat("Low-expression cutoff on the log2(TPM+0.001) scale:", round(CUTOFF, 5), "\n")

n_genes <- nrow(dt)
BATCH <- 3000   # genes per batch; lower this (e.g. to 1000) if it still runs out of memory
frac_silent <- numeric(n_genes)
gene_mad <- numeric(n_genes)

cat("Computing low-expression fraction and MAD in batches of", BATCH, "genes...\n")
for (start in seq(1, n_genes, by = BATCH)) {
  end <- min(start + BATCH - 1, n_genes)
  sub_mat <- as.matrix(dt[start:end, ..sample_cols])
  storage.mode(sub_mat) <- "numeric"
  frac_silent[start:end] <- rowMeans(sub_mat < CUTOFF)
  gene_mad[start:end] <- rowMads(sub_mat)
  rm(sub_mat)
  if (start %% (BATCH * 5) == 1) { gc(); cat("  processed genes", start, "-", end, "of", n_genes, "\n") }
}
cat("Finished computing per-gene statistics.\n")

keep_expr <- frac_silent < 0.90
cat(sum(keep_expr), "of", n_genes, "genes pass the low-expression filter.\n")

# ---- 5. Rank surviving genes by MAD, pick top 3000 / top 10000 (by ROW INDEX) ----
kept_indices <- which(keep_expr)
ord_within_kept <- order(gene_mad[kept_indices], decreasing = TRUE)
ranked_indices <- kept_indices[ord_within_kept]   # global row indices, MAD descending

top3000_idx  <- ranked_indices[1:3000]
top10000_idx <- ranked_indices[1:10000]

# ---- 6. NOW extract the actual expression data, but ONLY for the (small)
#         set of genes we're keeping - safe to materialize as a full matrix ----
top10000_dt <- dt[top10000_idx, ]
gene_ensg_10000 <- top10000_dt[[id_col]]
gene_sym_10000  <- top10000_dt[[sym_col]]
mat_10000 <- as.matrix(top10000_dt[, ..sample_cols])
storage.mode(mat_10000) <- "numeric"
rownames(mat_10000) <- gene_ensg_10000

is_top3000 <- top10000_idx %in% top3000_idx   # top3000 is the first 3000 of top10000 by construction
mat_3000 <- mat_10000[is_top3000, , drop = FALSE]
gene_ensg_3000 <- gene_ensg_10000[is_top3000]

rm(dt, top10000_dt); gc()   # free the big table now - everything needed is in mat_10000/mat_3000
cat("Extracted final gene matrices: ", nrow(mat_3000), " and ", nrow(mat_10000), " genes.\n", sep = "")

# ---- 7. Sanity check against the EXISTING 5000-gene panel ----
overlap_3000 <- sum(gene_ensg_3000 %in% existing_5000_genes)
cat("SANITY CHECK: ", overlap_3000, " / 3000 genes in the new 3000-gene panel already ",
    "appear in the existing 5000-gene panel (expect this close to 3000).\n", sep = "")

# ---- 8. Z-score standardize using the FULL 19,131-sample mean/SD ----
zscore_mat <- function(mat) {
  mu  <- rowMeans(mat)
  sdv <- apply(mat, 1, sd)
  sdv[sdv == 0] <- NA
  z <- sweep(mat, 1, mu, "-")
  z <- sweep(z, 1, sdv, "/")
  z
}

z_3000  <- zscore_mat(mat_3000)
z_10000 <- zscore_mat(mat_10000)

# ---- 9. Subset to the 835 malignant samples, in the SAME row order, and save ----
malig <- fread(file.path(base_dir, "Malignant_Subset_835.csv"))
malig_ids <- malig$sampleID
cat("Malignant_Subset_835.csv first 5 non-meta column names (for a quick format check):\n")
known_meta_cols <- c("sampleID", "sample_type", "Final_Label", "Final_Label1",
                      "Final_Label2", "PAM50Call_RNAseq", "PAM50_mRNA_nature2012", "_cohort")
print(head(setdiff(names(malig), known_meta_cols), 5))

write_panel_csv <- function(z, malig_ids, final_label, out_path) {
  matched <- intersect(colnames(z), malig_ids)
  cat("Matched", length(matched), "of", length(malig_ids),
      "malignant sample IDs for", basename(out_path), "\n")
  sub <- z[, malig_ids, drop = FALSE]
  sub_t <- t(sub)
  out <- data.table(sampleID = malig_ids)
  out <- cbind(out, as.data.table(sub_t))
  out$Final_Label <- final_label
  fwrite(out, out_path)
  cat("Saved:", out_path, "-", nrow(out), "rows,", ncol(out) - 2, "genes\n")
}

write_panel_csv(z_3000,  malig_ids, malig$Final_Label,
                file.path(base_dir, "Malignant_Subset_835_3000genes.csv"))
write_panel_csv(z_10000, malig_ids, malig$Final_Label,
                file.path(base_dir, "Malignant_Subset_835_10000genes.csv"))

cat("\nDone. Please paste back everything this script printed, especially:\n",
    " - the sample-matching count in step 2\n",
    " - the sanity-check overlap in step 7\n",
    " - the 'first 5 non-meta column names' in step 9\n",
    " - the 'Matched ... malignant sample IDs' lines\n", sep = "")
