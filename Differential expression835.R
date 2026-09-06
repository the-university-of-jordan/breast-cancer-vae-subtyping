# --- Keep only expression columns that belong to the malignant-subset metadata;
#     the expression file evidently contains additional (e.g. GTEx normal) samples
#     outside this subset, which are expected here and simply dropped. ---
matched_cols <- colnames(expr) %in% meta[[sample_col]]
cat("Expression columns matching metadata:", sum(matched_cols), "of", ncol(expr), "\n")
cat("(remaining columns are non-malignant-subset samples in the raw expression file and will be dropped)\n\n")

expr <- expr[, matched_cols, drop = FALSE]

# --- The direction that must be perfect: every one of the 835 malignant-subset
#     samples in metadata MUST have a matching expression column. ---
missing_from_expr <- setdiff(meta[[sample_col]], colnames(expr))
if (length(missing_from_expr) > 0) {
  stop(length(missing_from_expr), " malignant-subset sample(s) in metadata have NO matching column in the expression file. ",
       "Example missing ID: ", missing_from_expr[1])
}

# --- Reorder metadata to match the (now subsetted) expression column order ---
match_idx <- match(colnames(expr), meta[[sample_col]])
meta <- meta[match_idx, ]
stopifnot(identical(colnames(expr), meta[[sample_col]]))
stopifnot(nrow(meta) == 835)

cat("Aligned expression matrix:", nrow(expr), "genes x", ncol(expr), "samples (should be 835)\n\n")
cat("Cluster sizes (VAE Z=10, k=5):\n")
print(table(meta[[cluster_col]]))
cat("\n")