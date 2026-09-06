# ============================================================
# Section 6.2 (revised): GO (BP) + KEGG over-representation, and GSEA
# (Hallmark), per VAE Z=10 cluster (Hierarchical, k=5).
# Changes from the previous version:
#  - LOGFC_CUTOFF relaxed 1.0 -> 0.5 (this comparison's real effect
#    sizes are modest - cluster-vs-cluster, not tumor-vs-normal)
#  - enrichGO/enrichKEGG no longer hard-filter to p<0.05 internally;
#    the full ranked term table is saved, since the goal is "top 10-20
#    terms," not "only significant terms" - p.adjust is kept in the
#    output so real significance can still be checked per term.
# ============================================================

pkgs <- c("clusterProfiler", "org.Hs.eg.db", "msigdbr", "fgsea", "dplyr")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) {
    if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
    BiocManager::install(p, update = FALSE, ask = FALSE)
  }
}
library(clusterProfiler)
library(org.Hs.eg.db)
library(msigdbr)
library(fgsea)
library(dplyr)

base_dir <- "C:/Users/user/Desktop/Bioinformatics_Data/8749 samples"

PADJ_CUTOFF  <- 0.05
LOGFC_CUTOFF <- 0.5   # relaxed from 1.0 - see note above

clusters <- 1:5

de1 <- read.csv(file.path(base_dir, "DE_VAE_Z10_Cluster1_vs_Rest.csv"), stringsAsFactors = FALSE)
all_genes <- de1$gene_id
cat("Total genes in panel:", length(all_genes), "\n")

id_map <- bitr(all_genes, fromType = "ENSEMBL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)
id_map <- id_map[!duplicated(id_map$ENSEMBL), ]
cat("ENSEMBL -> ENTREZ mapped:", nrow(id_map), "of", length(all_genes),
    "(", round(100 * nrow(id_map) / length(all_genes), 1), "%)\n\n")

universe_ensembl <- all_genes
universe_entrez  <- unique(id_map$ENTREZID)

hallmark <- msigdbr(species = "Homo sapiens", category = "H")
hallmark_list <- split(hallmark$entrez_gene, hallmark$gs_name)
cat("Hallmark gene sets loaded:", length(hallmark_list), "\n\n")

for (cl in clusters) {
  cat("=====", "Cluster", cl, "=====\n")
  de <- read.csv(file.path(base_dir, paste0("DE_VAE_Z10_Cluster", cl, "_vs_Rest.csv")), stringsAsFactors = FALSE)
  
  mapped <- de %>%
    left_join(id_map, by = c("gene_id" = "ENSEMBL")) %>%
    rename(Ensembl_ID = gene_id, Gene_Symbol = gene_symbol) %>%
    select(Ensembl_ID, Gene_Symbol, ENTREZID, logFC, t, adj.P.Val) %>%
    arrange(desc(abs(t)))
  write.csv(mapped, file.path(base_dir, paste0("Cluster", cl, "_mapped_genes.csv")), row.names = FALSE)
  
  sig <- de %>% filter(adj.P.Val < PADJ_CUTOFF, abs(logFC) >= LOGFC_CUTOFF)
  cat("Significant genes (adj.P.Val <", PADJ_CUTOFF, "& |logFC| >=", LOGFC_CUTOFF, "):", nrow(sig), "\n")
  
  if (nrow(sig) == 0) {
    cat("No genes pass the threshold for this cluster - skipping GO/KEGG.\n\n")
    next
  }
  
  # ---- Cluster_n_GO_BP.csv: full ranked term table, top 10-20 taken when reporting ----
  go_res <- enrichGO(gene          = sig$gene_id,
                     universe      = universe_ensembl,
                     OrgDb         = org.Hs.eg.db,
                     keyType       = "ENSEMBL",
                     ont           = "BP",
                     pAdjustMethod = "BH",
                     pvalueCutoff  = 1,
                     qvalueCutoff  = 1)
  go_df <- as.data.frame(go_res) %>% arrange(p.adjust)
  write.csv(go_df, file.path(base_dir, paste0("Cluster", cl, "_GO_BP.csv")), row.names = FALSE)
  cat("GO BP terms tested:", nrow(go_df),
      " | reaching p.adjust<0.05:", sum(go_df$p.adjust < 0.05, na.rm = TRUE), "-> saved\n")
  
  # ---- Cluster_n_KEGG.csv ----
  sig_entrez <- id_map$ENTREZID[id_map$ENSEMBL %in% sig$gene_id]
  kegg_res <- enrichKEGG(gene          = sig_entrez,
                         universe      = universe_entrez,
                         organism      = "hsa",
                         pAdjustMethod = "BH",
                         pvalueCutoff  = 1,
                         qvalueCutoff  = 1)
  kegg_df <- as.data.frame(kegg_res) %>% arrange(p.adjust)
  write.csv(kegg_df, file.path(base_dir, paste0("Cluster", cl, "_KEGG.csv")), row.names = FALSE)
  cat("KEGG pathways tested:", nrow(kegg_df),
      " | reaching p.adjust<0.05:", sum(kegg_df$p.adjust < 0.05, na.rm = TRUE), "-> saved\n")
  
  # ---- Cluster_n_GSEA_Hallmark.csv (unchanged - already working well) ----
  de_entrez <- de %>%
    inner_join(id_map, by = c("gene_id" = "ENSEMBL")) %>%
    distinct(ENTREZID, .keep_all = TRUE) %>%
    arrange(desc(t))
  ranks <- de_entrez$t
  names(ranks) <- de_entrez$ENTREZID
  
  set.seed(42)
  gsea_res <- fgseaMultilevel(pathways = hallmark_list, stats = ranks, eps = 0)
  gsea_df <- as.data.frame(gsea_res) %>% arrange(padj)
  gsea_df$leadingEdge <- sapply(gsea_df$leadingEdge, paste, collapse = "/")
  write.csv(gsea_df, file.path(base_dir, paste0("Cluster", cl, "_GSEA_Hallmark.csv")), row.names = FALSE)
  cat("GSEA Hallmark pathways (padj<0.05):", sum(gsea_df$padj < 0.05, na.rm = TRUE), "-> saved\n\n")
}

cat("Section 6.2 complete.\n")