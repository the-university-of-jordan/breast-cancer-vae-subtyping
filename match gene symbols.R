if (!requireNamespace("writexl", quietly = TRUE)) install.packages("writexl")
library(writexl)

base_dir <- "C:/Users/user/Desktop/Bioinformatics_Data/8749 samples"

top10_list <- list()

for (cl in 1:5) {
  path <- file.path(base_dir, paste0("DE_VAE_Z10_Cluster", cl, "_vs_Rest.csv"))
  de <- read.csv(path, stringsAsFactors = FALSE)
  
  # Re-enforce rank order (|t| descending) defensively, then take the top 10.
  de <- de[order(-abs(de$t)), ]
  top10 <- de[1:10, c("gene_symbol", "logFC", "adj.P.Val")]
  top10$Cluster <- cl
  names(top10)[1] <- "Gene_Symbol"
  
  top10_list[[cl]] <- top10
  cat("Cluster", cl, "top 10 genes:", paste(top10$Gene_Symbol, collapse = ", "), "\n")
}

top10_all <- do.call(rbind, top10_list)
top10_all <- top10_all[, c("Gene_Symbol", "logFC", "adj.P.Val", "Cluster")]

out_path <- file.path(base_dir, "Top10_DriverGenes_VAE_Z10_AllClusters.xlsx")
write_xlsx(top10_all, out_path)

cat("\nSaved:", out_path, "(50 rows: 10 genes x 5 clusters)\n")