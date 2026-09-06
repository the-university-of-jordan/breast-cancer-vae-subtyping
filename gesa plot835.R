if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
library(ggplot2)
library(dplyr)

base_dir <- "C:/Users/user/Desktop/Bioinformatics_Data/8749 samples"
clusters <- 1:5

# --- Read the FULL Hallmark GSEA table per cluster (all 50 sets, not top 20) ---
all_gsea <- list()
for (cl in clusters) {
  df <- read.csv(file.path(base_dir, paste0("Cluster", cl, "_GSEA_Hallmark.csv")), stringsAsFactors = FALSE)
  df$Cluster <- cl
  all_gsea[[cl]] <- df[, c("Cluster", "pathway", "NES", "padj")]
}
gsea_all <- bind_rows(all_gsea)
cat("Total rows (5 clusters x Hallmark sets):", nrow(gsea_all), "\n")

# --- Keep pathways significant (padj < 0.05) in at least one cluster ---
sig_pathways <- gsea_all %>% filter(padj < 0.05) %>% pull(pathway) %>% unique()
cat("Pathways significant in >=1 cluster:", length(sig_pathways), "\n")
plot_df <- gsea_all %>% filter(pathway %in% sig_pathways)

# --- Assign each pathway to a biological theme, matching Section 3.6 / Table 16 ---
theme_map <- c(
  HALLMARK_E2F_TARGETS = "Proliferation", HALLMARK_G2M_CHECKPOINT = "Proliferation",
  HALLMARK_MITOTIC_SPINDLE = "Proliferation", HALLMARK_MYC_TARGETS_V1 = "Proliferation",
  HALLMARK_MYC_TARGETS_V2 = "Proliferation", HALLMARK_MTORC1_SIGNALING = "Proliferation",
  HALLMARK_ESTROGEN_RESPONSE_EARLY = "Hormone", HALLMARK_ESTROGEN_RESPONSE_LATE = "Hormone",
  HALLMARK_ANDROGEN_RESPONSE = "Hormone",
  HALLMARK_ALLOGRAFT_REJECTION = "Immune", HALLMARK_INTERFERON_GAMMA_RESPONSE = "Immune",
  HALLMARK_INTERFERON_ALPHA_RESPONSE = "Immune", HALLMARK_IL6_JAK_STAT3_SIGNALING = "Immune",
  HALLMARK_TNFA_SIGNALING_VIA_NFKB = "Immune", HALLMARK_INFLAMMATORY_RESPONSE = "Immune",
  HALLMARK_COMPLEMENT = "Immune", HALLMARK_IL2_STAT5_SIGNALING = "Immune",
  HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION = "EMT/Stromal", HALLMARK_ADIPOGENESIS = "EMT/Stromal",
  HALLMARK_MYOGENESIS = "EMT/Stromal", HALLMARK_COAGULATION = "EMT/Stromal",
  HALLMARK_ANGIOGENESIS = "EMT/Stromal", HALLMARK_APICAL_JUNCTION = "EMT/Stromal",
  HALLMARK_APICAL_SURFACE = "EMT/Stromal", HALLMARK_DNA_REPAIR = "DNA Repair"
)
plot_df$Theme <- ifelse(plot_df$pathway %in% names(theme_map), theme_map[plot_df$pathway], "Metabolic/Other")
plot_df$Label <- gsub("_", " ", gsub("HALLMARK_", "", plot_df$pathway))

# --- Order pathways within each theme by mean |NES| (strongest first) ---
order_df <- plot_df %>% group_by(Label) %>% summarise(m = mean(abs(NES)), Theme = first(Theme)) %>%
  arrange(Theme, desc(m))
plot_df$Label <- factor(plot_df$Label, levels = rev(order_df$Label))
plot_df$Cluster <- factor(plot_df$Cluster, levels = clusters, labels = paste("Cluster", clusters))

p <- ggplot(plot_df, aes(x = Cluster, y = Label, size = -log10(padj + 1e-300), color = NES)) +
  geom_point() +
  scale_color_gradient2(low = "#2166AC", mid = "white", high = "#B2182B", midpoint = 0, name = "NES") +
  scale_size_continuous(name = "-log10(padj)", range = c(1, 8)) +
  facet_grid(Theme ~ ., scales = "free_y", space = "free_y") +
  theme_minimal(base_size = 12) +
  theme(strip.text.y = element_text(angle = 0, face = "bold"),
        panel.spacing = unit(0.5, "lines"),
        axis.text.y = element_text(size = 9)) +
  labs(x = NULL, y = NULL, title = "Hallmark Gene Set Enrichment by Cluster\n(VAE Z=10, Hierarchical k=5)")

out_path <- file.path(base_dir, "Figure_Hallmark_GSEA_Dotplot.png")
ggsave(out_path, p, width = 9, height = 10, dpi = 300)
cat("Saved:", out_path, "\n")