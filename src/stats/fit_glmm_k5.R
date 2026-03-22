suppressPackageStartupMessages({
  library(lme4)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript src/stats/fit_glmm_k5.R <glmm_dataset_k5.csv> <glmm_results_k5.csv> <out_plot.png>")
}

in_csv <- args[1]
out_csv <- args[2]
out_png <- args[3]

df <- read.csv(in_csv, stringsAsFactors = FALSE)

required <- c("y_success", "caregiver_type", "child_id", "session_id")
missing <- setdiff(required, colnames(df))
if (length(missing) > 0) stop(paste("Missing columns:", paste(missing, collapse = ", ")))

df$y_success <- as.integer(df$y_success)
df$caregiver_type <- factor(df$caregiver_type, levels = c("MOT", "FAT"))
df$child_id <- factor(df$child_id)
df$session_id <- factor(df$session_id)

m <- glmer(
  y_success ~ caregiver_type + (1 | child_id) + (1 | session_id),
  data = df,
  family = binomial(link = "logit")
)

sm <- summary(m)

beta <- fixef(m)["caregiver_typeFAT"]
se <- sqrt(vcov(m)["caregiver_typeFAT", "caregiver_typeFAT"])

or <- exp(beta)
ci <- exp(beta + c(-1, 1) * 1.96 * se)

z <- beta / se
p <- 2 * (1 - pnorm(abs(z)))

out <- data.frame(
  term = "FAT_vs_MOT",
  beta = beta,
  odds_ratio = or,
  ci_low = ci[1],
  ci_high = ci[2],
  z = z,
  p_value = p
)

write.csv(out, out_csv, row.names = FALSE)

p_plot <- ggplot(out, aes(x = term, y = odds_ratio)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.15) +
  geom_hline(yintercept = 1.0, linetype = "dashed") +
  scale_y_log10() +
  labs(
    x = NULL,
    y = "Odds ratio (log scale)",
    title = "Caregiver effect on next child success (FAT vs MOT)"
  ) +
  theme_minimal(base_size = 12)

ggsave(out_png, p_plot, width = 7, height = 4, dpi = 200)

cat("Done.\n")
cat("Model summary:\n")
print(sm)
cat("\nSaved results to:", out_csv, "\n")
cat("Saved plot to:", out_png, "\n")
