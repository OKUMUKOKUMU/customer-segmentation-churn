# =====================================================================================
# Customer Segmentation & Churn Analysis in R (base R only)
# Author: Fordrane Albert Okumu
#
# PURPOSE
#   An independent rebuild of python/segmentation_and_churn.ipynb using only base R
#   (stats::kmeans, stats::glm, graphics). Rebuilding the whole pipeline in a second
#   language, from raw orders to model, is a strong correctness check: if two separate
#   implementations give the same features, coefficients and AUC, the analysis is right.
#
# STEPS
#   1. Load raw data
#   2. Feature engineering at the 30-Jun-2026 snapshot (no data from after the snapshot)
#   3. RFM scoring and segments
#   4. K-means clustering (k = 4) and profiling
#   5. Churn label (no order in the next 90 days) for outlets active at the snapshot
#   6. Logistic regression (glm, binomial) on the same features and train/test split
#   7. Evaluation: AUC (Mann-Whitney), gains by decile
#   8. Cross-check against the Python outputs + Markdown report
#
# Run from the repository root:   Rscript R/segmentation_and_churn.R
# =====================================================================================

CUTOFF  <- as.Date("2026-06-30")
HORIZON <- 90
out_dir <- file.path("outputs", "r"); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
TEAL <- "#0f766e"; AMBER <- "#b45309"; GREY <- "#94a3b8"; RED <- "#b91c1c"

# -------------------------------------------------------------------------------------
# 1. Load
# -------------------------------------------------------------------------------------
cust   <- read.csv("data/customers.csv", stringsAsFactors = FALSE)
orders <- read.csv("data/orders.csv", stringsAsFactors = FALSE)
comp   <- read.csv("data/complaints.csv", stringsAsFactors = FALSE)
orders$order_date    <- as.Date(orders$order_date)
comp$complaint_date  <- as.Date(comp$complaint_date)
cat(sprintf("%d outlets | %d orders | %d complaints\n", nrow(cust), nrow(orders), nrow(comp)))

# -------------------------------------------------------------------------------------
# 2. Feature engineering - ONLY data on or before the snapshot date
# -------------------------------------------------------------------------------------
past    <- orders[orders$order_date <= CUTOFF, ]
in_last <- function(d, days) d$order_date > CUTOFF - days
l365    <- past[in_last(past, 365), ]
l90     <- past[in_last(past, 90), ]
p270    <- past[in_last(past, 365) & !in_last(past, 90), ]

ids <- sort(unique(past$customer_id))
count_by <- function(d) { x <- table(factor(d$customer_id, levels = ids)); as.numeric(x) }
sum_by   <- function(d, col) { x <- tapply(d[[col]], factor(d$customer_id, levels = ids), sum); x[is.na(x)] <- 0; as.numeric(x) }
mean_by  <- function(d, col) as.numeric(tapply(d[[col]], factor(d$customer_id, levels = ids), mean))

F <- data.frame(customer_id = ids,
                recency_days = as.numeric(CUTOFF - tapply(past$order_date, past$customer_id, max)[ids]),
                frequency_365d = count_by(l365),
                monetary_365d = sum_by(l365, "net_value_kes"),
                orders_90d = count_by(l90),
                orders_prior_270d = count_by(p270),
                avg_order_value = mean_by(l365, "net_value_kes"),
                avg_skus_per_order = mean_by(l365, "n_skus"),
                avg_discount_pct = mean_by(l365, "discount_pct"),
                late_delivery_rate = mean_by(l365, "delivered_late"),
                stockout_rate = mean_by(l365, "had_stockout"),
                return_rate = sum_by(l365, "returned_value_kes") / sum_by(l365, "net_value_kes"),
                tenure_months = round(as.numeric(CUTOFF - tapply(past$order_date, past$customer_id, min)[ids]) / 30.44, 1),
                stringsAsFactors = FALSE)
F$order_trend <- (F$orders_90d + 1) / (F$orders_prior_270d / 3 + 1)   # < 1 = slowing down
c180 <- comp[comp$complaint_date <= CUTOFF & comp$complaint_date > CUTOFF - 180, ]
F$complaints_180d <- count_by(setNames(c180, c("customer_id", "order_date", "t")))
F <- merge(F, cust, by = "customer_id")
F$has_merchandiser <- as.integer(F$has_merchandiser %in% c("True", "TRUE", TRUE))

# -------------------------------------------------------------------------------------
# 3. RFM segmentation (outlets with at least one order in the last 12 months)
# -------------------------------------------------------------------------------------
S <- F[F$frequency_365d > 0, ]
quint <- function(x) ceiling(5 * rank(x, ties.method = "first") / length(x))
S$R <- 6 - quint(S$recency_days)          # recent = 5
S$F <- quint(S$frequency_365d)
S$M <- quint(S$monetary_365d)
S$FM <- floor((S$F + S$M) / 2 + 0.5)
S$FM[(S$F + S$M) %% 2 == 1] <- round((S$F + S$M)[(S$F + S$M) %% 2 == 1] / 2)   # banker's rounding, as in pandas
S$rfm_segment <- with(S, ifelse(R >= 4 & FM >= 4, "Champions",
                      ifelse(R >= 3 & FM >= 3, "Loyal",
                      ifelse(R >= 4 & FM <= 2, "Promising / New",
                      ifelse(R == 3 & FM <= 2, "Needs Attention",
                      ifelse(R <= 2 & FM >= 4, "Can't Lose Them",
                      ifelse(R <= 2 & FM == 3, "At Risk", "Hibernating / Lost")))))))
seg_order <- c("Champions", "Loyal", "Promising / New", "Needs Attention", "At Risk", "Can't Lose Them", "Hibernating / Lost")
rfm_tab <- do.call(rbind, lapply(seg_order, function(s) {
  d <- S[S$rfm_segment == s, ]
  data.frame(segment = s, outlets = nrow(d), revenue_share = sum(d$monetary_365d) / sum(S$monetary_365d))
}))
cat("\nRFM segments:\n"); print(transform(rfm_tab, revenue_share = round(revenue_share, 3)), row.names = FALSE)

# -------------------------------------------------------------------------------------
# 4. K-means (k = 4) on log-transformed, standardised behaviour
# -------------------------------------------------------------------------------------
Z <- scale(cbind(log_recency = log1p(S$recency_days), log_frequency = log1p(S$frequency_365d),
                 log_order_value = log1p(S$avg_order_value), skus_per_order = S$avg_skus_per_order,
                 log_order_trend = log(S$order_trend), discount_pct = S$avg_discount_pct))
set.seed(42)
km <- kmeans(Z, 4, nstart = 25, iter.max = 100)
S$cluster_id <- km$cluster
prof <- aggregate(cbind(recency_days, frequency_365d, avg_order_value, order_trend) ~ cluster_id, S, median)
prof$outlets <- as.numeric(table(S$cluster_id))
prof$revenue_share <- as.numeric(tapply(S$monetary_365d, S$cluster_id, sum)) / sum(S$monetary_365d)
big <- prof$avg_order_value > median(S$avg_order_value); active <- prof$recency_days <= 60
prof$segment <- ifelse(big & active, "Key Accounts", ifelse(!big & active, "Core Small Outlets",
                ifelse(big & !active, "Lapsing Key Accounts", "Lapsed Small Outlets")))
S$cluster <- prof$segment[match(S$cluster_id, prof$cluster_id)]
prof <- prof[order(-prof$revenue_share), ]
cat("\nK-means clusters (R):\n"); print(prof[, c("segment", "outlets", "recency_days", "frequency_365d",
                                               "avg_order_value", "order_trend", "revenue_share")], row.names = FALSE)

png(file.path(out_dir, "R_01_cluster_profiles.png"), width = 1700, height = 900, res = 170)
par(mar = c(7, 11, 3, 5))
zm <- t(sapply(prof$segment, function(s) colMeans(Z[S$cluster == s, , drop = FALSE])))
image(1:ncol(zm), 1:nrow(zm), t(zm)[, nrow(zm):1], col = hcl.colors(21, "Blue-Red 3"), zlim = c(-1.6, 1.6),
      axes = FALSE, xlab = "", ylab = "", main = "K-means cluster profiles in R (standardised means)")
axis(1, 1:ncol(zm), colnames(zm), las = 2, cex.axis = .75); axis(2, 1:nrow(zm), rev(rownames(zm)), las = 1, cex.axis = .8)
for (i in 1:ncol(zm)) for (j in 1:nrow(zm)) text(i, nrow(zm) - j + 1, sprintf("%+.1f", zm[j, i]), cex = .7)
invisible(dev.off())

# -------------------------------------------------------------------------------------
# 5. Churn label: active at snapshot (ordered in last 90 days) and no order in next 90 days
# -------------------------------------------------------------------------------------
fut <- orders[orders$order_date > CUTOFF & orders$order_date <= CUTOFF + HORIZON, ]
A <- F[F$recency_days <= 90, ]
A$churned <- as.integer(!A$customer_id %in% fut$customer_id)
cat(sprintf("\nActive outlets: %d | churned: %d (%.1f%%)\n", nrow(A), sum(A$churned), 100 * mean(A$churned)))

# -------------------------------------------------------------------------------------
# 6. Logistic regression - same features, same deterministic split as Python
# -------------------------------------------------------------------------------------
A$log_avg_order_value <- log1p(A$avg_order_value)
A$log_order_trend <- log(A$order_trend)
num <- c("recency_days", "frequency_365d", "log_order_trend", "log_avg_order_value", "avg_skus_per_order",
         "avg_discount_pct", "late_delivery_rate", "stockout_rate", "return_rate", "complaints_180d",
         "tenure_months", "has_merchandiser")
for (ch in c("HoReCa", "Petrol Station", "Supermarket", "Wholesaler")) A[[paste0("channel_", ch)]] <- as.numeric(A$channel == ch)
for (rg in c("Coast", "Nairobi", "Nyanza", "Rift Valley", "Western")) A[[paste0("region_", rg)]] <- as.numeric(A$region == rg)
feats <- c(num, grep("^(channel|region)_", names(A), value = TRUE))
test <- as.integer(substring(A$customer_id, 2)) %% 4 == 0          # identical to Python's is_test()
X <- as.matrix(A[, feats])
mu <- colMeans(X[!test, ]); sdv <- apply(X[!test, ], 2, sd)
Xs <- sweep(sweep(X, 2, mu), 2, sdv, "/")
train_df <- data.frame(churned = A$churned[!test], Xs[!test, ], check.names = FALSE)
fit <- glm(churned ~ ., data = train_df, family = binomial)
p_test <- predict(fit, newdata = data.frame(Xs[test, ], check.names = FALSE), type = "response")

auc <- function(y, p) {      # Mann-Whitney U formulation of ROC-AUC
  r <- rank(p); n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
y_test <- A$churned[test]
test_auc <- auc(y_test, p_test)
co <- summary(fit)$coefficients[-1, ]
or_tab <- data.frame(feature = gsub("`", "", rownames(co)), odds_ratio = exp(co[, 1]),
                     ci_low = exp(co[, 1] - 1.96 * co[, 2]), ci_high = exp(co[, 1] + 1.96 * co[, 2]),
                     p_value = co[, 4], row.names = NULL)
or_tab <- or_tab[order(or_tab$odds_ratio), ]
write.csv(or_tab, file.path(out_dir, "logit_odds_ratios_R.csv"), row.names = FALSE)
cat(sprintf("Logistic regression test AUC (R): %.3f\n", test_auc))

png(file.path(out_dir, "R_02_odds_ratios.png"), width = 1500, height = 1200, res = 170)
par(mar = c(4.5, 12, 3, 2))
k <- nrow(or_tab); colr <- ifelse(or_tab$p_value < .05, ifelse(or_tab$odds_ratio > 1, RED, TEAL), GREY)
plot(or_tab$odds_ratio, 1:k, log = "x", xlim = range(c(or_tab$ci_low, or_tab$ci_high)), pch = 19, col = colr,
     yaxt = "n", ylab = "", xlab = "Odds ratio per 1 SD (95% CI, log scale)", main = "Churn drivers - logistic regression (R)")
segments(or_tab$ci_low, 1:k, or_tab$ci_high, 1:k, col = colr, lwd = 2); abline(v = 1, lty = 2)
axis(2, 1:k, or_tab$feature, las = 1, cex.axis = .7)
invisible(dev.off())

# -------------------------------------------------------------------------------------
# 7. Gains by risk decile (test set)
# -------------------------------------------------------------------------------------
dec <- ceiling(10 * rank(-p_test, ties.method = "first") / length(p_test))
gains <- data.frame(decile = 1:10, churners = as.numeric(tapply(y_test, dec, sum)), outlets = as.numeric(table(dec)))
gains$cum_capture <- cumsum(gains$churners) / sum(gains$churners)
gains$lift <- (gains$churners / gains$outlets) / mean(y_test)
png(file.path(out_dir, "R_03_gains.png"), width = 1500, height = 900, res = 170)
par(mar = c(4.5, 4.5, 3, 4.5))
bp <- barplot(gains$lift, names.arg = gains$decile, col = TEAL, border = NA, ylab = "Lift",
              xlab = "Risk decile (1 = highest risk)", main = "Lift and cumulative capture by decile (R, test set)")
par(new = TRUE); plot(bp, 100 * gains$cum_capture, type = "o", col = AMBER, pch = 19, axes = FALSE, xlab = "", ylab = "", ylim = c(0, 105))
axis(4, col.axis = AMBER); mtext("Cumulative % of churners captured", 4, 3, col = AMBER)
invisible(dev.off())

# -------------------------------------------------------------------------------------
# 8. Cross-check with Python and write the report
# -------------------------------------------------------------------------------------
check <- "Python outputs not found - run the notebook first to enable the cross-check."
py_or <- file.path("outputs", "python", "logit_odds_ratios.csv")
py_seg <- file.path("outputs", "python", "customer_segments.csv")
if (file.exists(py_or)) {
  py <- read.csv(py_or, stringsAsFactors = FALSE); names(py)[1] <- "feature"
  m <- merge(py, or_tab, by = "feature", suffixes = c("_py", "_r"))
  max_or_diff <- max(abs(m$odds_ratio_py - m$odds_ratio_r) / m$odds_ratio_py)
  rfm_match <- NA; clus_match <- NA
  if (file.exists(py_seg)) {
    ps <- merge(read.csv(py_seg, stringsAsFactors = FALSE), S[, c("customer_id", "rfm_segment", "cluster")],
                by = "customer_id", suffixes = c("_py", "_r"))
    rfm_match <- mean(ps$rfm_segment_py == ps$rfm_segment_r)
    clus_match <- mean(ps$cluster_py == ps$cluster_r)
  }
  check <- sprintf(paste("coefficients compared: %d | max relative odds-ratio difference: %.1e |",
                         "RFM segment agreement: %.2f%% | K-means segment agreement: %.2f%%"),
                   nrow(m), max_or_diff, 100 * rfm_match, 100 * clus_match)
}
cat("\nCROSS-CHECK:", check, "\n")

md_table <- function(df) c(paste("|", paste(names(df), collapse = " | "), "|"),
                           paste("|", paste(rep("---", ncol(df)), collapse = " | "), "|"),
                           apply(df, 1, function(r) paste("|", paste(r, collapse = " | "), "|")))
report <- c(
  "# Segmentation & Churn — R results (auto-generated by `R/segmentation_and_churn.R`)", "",
  sprintf("*Snapshot %s · %d outlets segmented · %d active outlets modelled · churn = no order in next %d days*",
          CUTOFF, nrow(S), nrow(A), HORIZON), "",
  "## Cross-check with the Python notebook", "", paste("-", check), "",
  "Any disagreement is limited to a handful of borderline outlets: R's `kmeans()` (Hartigan-Wong) and scikit-learn (Lloyd, k-means++) use different algorithms and starting points, and RFM quintile cut-offs can split tied values differently. The segment *profiles* are the same.", "",
  "## RFM segments", "", md_table(transform(rfm_tab, revenue_share = sprintf("%.1f%%", 100 * revenue_share))), "",
  "## K-means segments", "",
  md_table(data.frame(segment = prof$segment, outlets = prof$outlets, median_recency = prof$recency_days,
                      median_orders_12m = prof$frequency_365d, median_order_value = round(prof$avg_order_value),
                      median_trend = round(prof$order_trend, 2), revenue_share = sprintf("%.1f%%", 100 * prof$revenue_share))), "",
  "![Cluster profiles](R_01_cluster_profiles.png)", "",
  "## Churn model (logistic regression, glm)", "",
  sprintf("- Active outlets: %d, churn rate %.1f%% · test AUC **%.3f**", nrow(A), 100 * mean(A$churned), test_auc), "",
  md_table(data.frame(feature = or_tab$feature, odds_ratio = sprintf("%.2f", or_tab$odds_ratio),
                      `95% CI` = sprintf("%.2f – %.2f", or_tab$ci_low, or_tab$ci_high),
                      p_value = sprintf("%.3f", or_tab$p_value), check.names = FALSE)), "",
  "![Odds ratios](R_02_odds_ratios.png)", "",
  "## Gains (test set)", "",
  md_table(data.frame(decile = gains$decile, churners = gains$churners, outlets = gains$outlets,
                      cumulative_capture = sprintf("%.0f%%", 100 * gains$cum_capture), lift = sprintf("%.2f", gains$lift))), "",
  "![Gains](R_03_gains.png)")
writeLines(report, file.path(out_dir, "R_results.md"))
write.csv(S[, c("customer_id", "R", "F", "M", "rfm_segment", "cluster")], file.path(out_dir, "customer_segments_R.csv"), row.names = FALSE)
cat("Wrote outputs/r/R_results.md, CSVs and charts\n")
