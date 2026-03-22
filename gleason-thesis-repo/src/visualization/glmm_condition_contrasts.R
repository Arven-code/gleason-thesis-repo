suppressPackageStartupMessages({
  library(lme4); library(dplyr); library(readr); library(ggplot2); library(tidyr)
})

artifacts <- "artifacts"
paths <- c(
  Prosody = file.path(artifacts, "train_rows_for_glmm_prosody.csv"),
  `Text-only` = file.path(artifacts, "train_rows_for_glmm_textonly.csv"),
  `Shuffled PROS` = file.path(artifacts, "train_rows_for_glmm_shufflePROS.csv")
)

dfs <- list()
for (nm in names(paths)) {
  p <- paths[[nm]]
  if (!file.exists(p)) {
    warning("Missing file: ", p)
    next
  }
  df <- read_csv(p, show_col_types = FALSE)
  df$condition <- nm
  dfs[[nm]] <- df
}
if (length(dfs) == 0) stop("No input CSVs found in artifacts/")

dat <- bind_rows(dfs)

grp_col <- if ("child_id" %in% names(dat)) "child_id" else
           if ("__source_file" %in% names(dat)) "__source_file" else NULL
if (is.null(grp_col)) stop("No child_id or __source_file columns present.")
if (!"label_acc" %in% names(dat)) stop("Missing label_acc in data.")

dat$grp_id <- as.factor(dat[[grp_col]])
dat$condition <- factor(dat$condition, levels = c("Text-only", "Prosody", "Shuffled PROS"))

m <- glmer(label_acc ~ condition + (1|grp_id),
           data = dat, family = binomial,
           control = glmerControl(optimizer = "bobyqa"))

sm <- summary(m)
co <- as.data.frame(sm$coefficients)
co$term <- rownames(co)
rownames(co) <- NULL

wald_ci <- function(est, se) {
  lo <- est - 1.96 * se
  hi <- est + 1.96 * se
  c(lo = lo, hi = hi)
}

int_row <- co %>% filter(term == "(Intercept)") %>% slice(1)
int_ci <- wald_ci(int_row$Estimate, int_row$`Std. Error`)
baseline_tbl <- tibble(
  contrast = "Baseline (Text-only)",
  log_odds = int_row$Estimate,
  se = int_row$`Std. Error`,
  z = int_row$`z value`,
  p = int_row$`Pr(>|z|)`,
  OR = exp(int_row$Estimate),
  OR_low = exp(int_ci["lo"]),
  OR_high = exp(int_ci["hi"])
)

cond_tbl <- co %>%
  filter(grepl("^condition", term)) %>%
  transmute(
    contrast = gsub("^condition", "", term) |> trimws(),
    log_odds = Estimate,
    se = `Std. Error`,
    z = `z value`,
    p = `Pr(>|z|)`
  ) %>%
  rowwise() %>%
  mutate(
    OR = exp(log_odds),
    OR_low = exp(log_odds - 1.96 * se),
    OR_high = exp(log_odds + 1.96 * se)
  ) %>%
  ungroup()

pred_tbl <- tibble(
  condition = c("Text-only", "Prosody", "Shuffled PROS"),
  log_odds = c(
    int_row$Estimate,
    int_row$Estimate + (cond_tbl %>% filter(contrast == "Prosody") %>% pull(log_odds) %>% {if(length(.) == 0) 0 else .}),
    int_row$Estimate + (cond_tbl %>% filter(contrast == "Shuffled PROS") %>% pull(log_odds) %>% {if(length(.) == 0) 0 else .})
  )
) %>%
  mutate(
    odds = exp(log_odds),
    prob = odds / (1 + odds)
  )

dir.create(artifacts, showWarnings = FALSE, recursive = TRUE)

effects_csv <- file.path(artifacts, "glmm_condition_effects.csv")
baseline_csv <- file.path(artifacts, "glmm_textonly_baseline.csv")
pred_csv <- file.path(artifacts, "glmm_predicted_prob_by_condition.csv")

write_csv(cond_tbl, effects_csv)
write_csv(baseline_tbl, baseline_csv)
write_csv(pred_tbl, pred_csv)

cat("Saved condition contrasts (vs Text-only) -> ", effects_csv, "\n")
cat("Saved Text-only baseline (intercept)     -> ", baseline_csv, "\n")
cat("Saved predicted prob by condition        -> ", pred_csv, "\n")

plot_df <- cond_tbl %>%
  mutate(contrast = gsub("^\\s+", "", contrast)) %>%
  mutate(contrast = factor(contrast, levels = c("Prosody", "Shuffled PROS")))

g <- ggplot(plot_df, aes(x = contrast, y = OR, ymin = OR_low, ymax = OR_high)) +
  geom_hline(yintercept = 1, linetype = 2) +
  geom_pointrange() +
  coord_flip() +
  scale_y_log10() +
  labs(
    title = "Condition effects vs Text-only (mixed logit, 95% CI)",
    x = NULL, y = "Odds Ratio (log scale)"
  ) +
  theme_minimal(base_size = 12)

plot_path <- file.path(artifacts, "glmm_condition_effects.png")
ggsave(plot_path, g, width = 7, height = 4, dpi = 200)
cat("Saved plot -> ", plot_path, "\n")
