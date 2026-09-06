## ---- 9. Read the GCT score output back in and reshape (samples as rows) ----
score_tab  <- read.delim(score_gct, skip = 2, check.names = FALSE, stringsAsFactors = FALSE)

score_names <- score_tab$NAME
score_vals  <- score_tab[, !(names(score_tab) %in% c("NAME", "Description")), drop = FALSE]
rownames(score_vals) <- score_names

scores_df <- as.data.frame(t(score_vals))

cat("Samples in ESTIMATE output:", nrow(scores_df), "\n")
cat("Samples in meta_835:", nrow(meta_835), "\n")
cat("Example of a possibly-mangled name from ESTIMATE output:", rownames(scores_df)[1], "\n")
cat("Corresponding original sample ID (meta_835, same position):", meta_835[[sample_col]][1], "\n")

stopifnot(nrow(scores_df) == nrow(meta_835))

## ---- 10. Reattach sample ID and cluster label by POSITION (order is preserved by ESTIMATE,
##          even though it mangles the actual ID strings), then save ----
scores_df[[sample_col]]  <- meta_835[[sample_col]]
scores_df[[cluster_col]] <- meta_835[[cluster_col]]

final_df <- scores_df[, c(sample_col, "StromalScore", "ImmuneScore", "ESTIMATEScore",
                          "TumorPurity", cluster_col)]

write.csv(final_df, out_csv, row.names = FALSE)
cat("\nSaved:", out_csv, "\n")
cat("Preview:\n")
print(head(final_df))