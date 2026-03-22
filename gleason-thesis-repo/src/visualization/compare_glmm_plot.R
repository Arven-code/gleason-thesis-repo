suppressPackageStartupMessages({
  library(lme4); library(dplyr); library(readr); library(ggplot2)
})

artifacts <- "artifacts"
files <- c(
  prosody  = file.path(artifacts, "train_rows_for_glmm_prosody.csv"),
  textonly = file.path(artifacts, "train_rows_for_glmm_textonly.csv"),
  shuffle  = file.path(artifacts, "train_rows_for_glmm_shufflePROS.csv")
)

fit_one <- function(path, label) {
  if (!file.exists(path)) { warning("Missing: ", path); return(NULL) }
  df <- read_csv(path, show_col_types = FALSE)

  id_col <- if ("child_id" %in% names(df)) "child_id" else
            if ("__source_file" %in% names(df)) "__source_file" else NULL
  if (is.null(id_col) || !"label_acc" %in% names(df)) { return(NULL) }

  df$grp_id <- as.factor(df[[id_col]])

  m <- tryCatch(
    glmer(label_acc ~ 1 + (1|grp_id), data=df,
          family=binomial, control=glmerControl(optimizer="bobyqa")),
    error=function(e) NULL
  )
  if (is.null(m)) return(NULL)

  co <- summary(m)$coefficients
  est <- co["(Intercept)", "Estimate"]
  se  <- co["(Intercept)", "Std. Error"]

  tibble(
    condition = label,
    term = "(Intercept)",
    estimate = est,
    std_error = se,
    or = exp(est),
    or_low = exp(est - 1.96 * se),
    or_high = exp(est + 1.96 * se)
  )
}

res <- bind_rows(
  fit_one(files["prosody"],  "Prosody"),
  fit_one(files["textonly"], "Text-only"),
  fit_one(files["shuffle"],  "Shuffled PROS")
)

if (nrow(res) == 0) stop("No model coefficients to plot (check input CSVs).")

print(res %>% select(condition, estimate, std_error, or, or_low, or_high))

dir.create(artifacts, showWarnings=FALSE, recursive=TRUE)
write_csv(res, file.path(artifacts, "glmm_coef_table.csv"))

p <- ggplot(res, aes(x=condition, y=or, ymin=or_low, ymax=or_high)) +
  geom_hline(yintercept=1, linetype=2) +
  geom_pointrange() +
  coord_flip() +
  scale_y_log10() +
  labs(
    title="Baseline odds of accurate utterance (mixed logit, 95% CI)",
    x=NULL,
    y="Odds ratio (log scale)"
  ) +
  theme_minimal(base_size=12)

ggsave(file.path(artifacts, "glmm_coef_forest.png"), p, width=7, height=4, dpi=200)
cat("Saved plot -> artifacts/glmm_coef_forest.png\n")
cat("Saved table -> artifacts/glmm_coef_table.csv\n")
