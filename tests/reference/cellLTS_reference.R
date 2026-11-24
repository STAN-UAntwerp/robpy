# Reference results of the authors' R code of cellLTS for robpy's tests.
#
# The code accompanies Raymaekers & Rousseeuw (2026), "Least trimmed squares regression with
# missing values and cellwise outliers" (arXiv:2603.04632), and can be downloaded from
#   https://wis.kuleuven.be/statdatascience/code/cellreg_code_script.zip
# Run from the repository root, after generate_inputs.py, with the path to cellLTS_code.R:
#
#   Rscript tests/reference/cellLTS_reference.R path/to/cellLTS_code.R
#
# Generated with R 4.5.1, cellWise 2.5.7 and robustbase 0.99.6. Besides the results, the
# script saves the random row orderings that randomDNsymmetrize draws (seeds 123 and 321),
# so that robpy can be run with exactly the same pairs of observations.

args <- commandArgs(TRUE)
suppressMessages(library(cellWise))
source(args[1])

here <- "tests/reference"
out_dir <- file.path(here, "expected")
dir.create(out_dir, showWarnings = FALSE)
write_out <- function(x, name) {
  x <- as.data.frame(x)
  num <- vapply(x, is.numeric, logical(1))
  x[num] <- lapply(x[num], signif, digits = 10)
  write.csv(x, file.path(out_dir, paste0(name, ".csv")), row.names = FALSE)
}

D <- as.matrix(read.csv(file.path(here, "data", "cell_lts.csv")))
X <- D[, 1:4]
y <- D[, 5]
Xnew <- as.matrix(read.csv(file.path(here, "data", "cell_lts_new.csv")))[, 1:4]

# the orderings of randomDNsymmetrize: each is a random permutation of the previous one
orders <- function(n, k, seed) {
  set.seed(seed, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
  o <- seq_len(n)
  out <- matrix(0L, k, n)
  for (i in seq_len(k)) {
    o <- o[sample(n)]
    out[i, ] <- o - 1L
  }
  out
}
write_out(orders(nrow(X), 20, 123), "cell_lts_orders_123")
write_out(orders(sum(!is.na(y)), 20, 321), "cell_lts_orders_321")

run <- function(case, ..., X = get("X", envir = globalenv()), Xnew = get("Xnew", envir = globalenv())) {
  out <- cellLTS(X, y, ...)
  # the raw LTS fit inside cellLTS (same inputs and seed), so that robpy can be checked with it
  pars <- list(...)
  observed <- !is.na(y)
  if (identical(pars$symmetrize, "none")) {
    lts <- ltsRidgeReg(y[observed], out$Ximp[observed, , drop = FALSE], 0.5, 1e-4, seed = 123)
  } else {
    sym <- randomDNsymmetrize(out$Ximp[observed, , drop = FALSE], y[observed], k = 20, seed = 321)
    lts <- ltsRidgeReg(sym$y, sym$X, 0.5, 1e-4, seed = 123)
  }
  write_out(lts$raw.coefficients, paste0(case, "_raw_lts"))
  write_out(out$coefficients, paste0(case, "_coefficients"))
  write_out(data.frame(sigma = out$sigma), paste0(case, "_scale"))
  write_out(out$mu_X, paste0(case, "_mu"))
  write_out(out$Sigma_X, paste0(case, "_covariance"))
  write_out(out$W_X, paste0(case, "_W"))
  write_out(out$Ximp, paste0(case, "_Ximp"))
  write_out(out$preds, paste0(case, "_preds"))
  write_out(out$v, paste0(case, "_v"))
  write_out(cellLTS_predict(out, Xnew), paste0(case, "_new_preds"))
}
run("cell_lts_default")
run("cell_lts_none", symmetrize = "none")
run("cell_lts_iterate", iterateMu = TRUE)

# one predictor: the first column (its differences use only the rows with an observed x)
write_out(orders(sum(!is.na(X[, 1])), 20, 123), "cell_lts_orders_123_univariate")
run("cell_lts_univariate", X = X[, 1, drop = FALSE], Xnew = Xnew[, 1, drop = FALSE])
