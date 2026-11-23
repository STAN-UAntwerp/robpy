# Reference results of the R package cellWise for robpy's tests.
#
# Run from the repository root, after generate_inputs.py:
#
#   Rscript tests/reference/cellwise_reference.R
#
# Generated with R 4.5.1 and cellWise 2.5.7. The results are written to
# tests/reference/expected. All functions are called with coreOnly = TRUE (no
# checkDataSet), except DDC fits that are followed by DDCpredict, which needs
# the checked data; checkDataSet leaves those data sets unchanged.

suppressMessages(library(cellWise))
stopifnot(packageVersion("cellWise") == "2.5.7")

here <- "tests/reference"
data_dir <- file.path(here, "data")
out_dir <- file.path(here, "expected")
dir.create(out_dir, showWarnings = FALSE)

read_data <- function(name) as.matrix(read.csv(file.path(data_dir, paste0(name, ".csv"))))
write_out <- function(x, name) {
  # 10 significant digits; the tests compare with a relative tolerance of 1e-8
  x <- as.data.frame(x)
  num <- vapply(x, is.numeric, logical(1))
  x[num] <- lapply(x[num], signif, digits = 10)
  write.csv(x, file.path(out_dir, paste0(name, ".csv")), row.names = FALSE)
}
derived_input <- function(x, name) {
  # write a derived input and read it back, so that R and robpy use the same numbers
  write_out(x, name)
  as.matrix(read.csv(file.path(out_dir, paste0(name, ".csv"))))
}
cellmask <- function(ind, X) {
  W <- matrix(0L, nrow(X), ncol(X))
  W[ind] <- 1L
  W
}

## univariate location and scale, and wrapping ---------------------------------

U <- read_data("univariate")
res <- NULL
for (type in c("1stepM", "hubhub", "wrap", "mcd", "rawmcd", "wrapmedmad")) {
  for (center in c(TRUE, FALSE)) {
    ls <- estLocScale(U, type = type, center = center, silent = TRUE)
    res <- rbind(res, data.frame(type = type, center = center, column = seq_len(ncol(U)),
                                 loc = ls$loc, scale = ls$scale))
  }
}
write_out(res, "loc_scale")
ls <- estLocScale(U[, 1:3], silent = TRUE)
write_out(wrap(U[, 1:3], ls$loc, ls$scale, checkPars = list(coreOnly = TRUE))$Xw, "wrap")

## DDC and DDCpredict ----------------------------------------------------------

run_ddc <- function(case, X, pars, n_fit = nrow(X)) {
  Xfit <- X[seq_len(n_fit), , drop = FALSE]
  predict_new <- n_fit < nrow(X)
  out <- DDC(Xfit, c(pars, list(silent = TRUE, coreOnly = !predict_new)))
  if (predict_new) stopifnot(identical(dim(out$remX), dim(Xfit)))
  write_out(cellmask(out$indcells, Xfit), paste0(case, "_flags"))
  write_out(out$stdResid, paste0(case, "_residuals"))
  write_out(out$Ximp, paste0(case, "_imputed"))
  write_out(data.frame(row = seq_len(n_fit) %in% out$indrows + 0L), paste0(case, "_rows"))
  if (predict_new) {
    Xnew <- X[(n_fit + 1):nrow(X), , drop = FALSE]
    new <- DDCpredict(Xnew, out)
    write_out(cellmask(new$indcells, Xnew), paste0(case, "_new_flags"))
    write_out(new$stdResid, paste0(case, "_new_residuals"))
    write_out(new$Ximp, paste0(case, "_new_imputed"))
    write_out(data.frame(row = seq_len(nrow(Xnew)) %in% new$indrows + 0L),
              paste0(case, "_new_rows"))
  }
}

D <- read_data("ddc_small")
run_ddc("ddc_default", D, list(), n_fit = 200)
D200 <- D[1:200, ]
run_ddc("ddc_hubhub", D200, list(tolProb = 0.999, standType = "hubhub", corrlim = 0.3))
run_ddc("ddc_wrap_cell09", D200, list(tolProbCell = 0.9, standType = "wrap"))
ls <- estLocScale(D200, silent = TRUE)
Ds <- derived_input(scale(D200, ls$loc, ls$scale), "ddc_center0_input")
run_ddc("ddc_center0", Ds, list(tolProbCell = 0.9, standType = "wrap", center = rep(0, ncol(D))))
run_ddc("ddc_neighbors", read_data("ddc_neighbors"), list(nbngbrs = 4), n_fit = 80)

## DDCWcov and cellMCD ---------------------------------------------------------

M <- read_data("cellmcd")
ls <- estLocScale(M, silent = TRUE)
Ms <- derived_input(scale(M, ls$loc, ls$scale), "ddcw_input")
ddcw <- cellWise:::DDCWcov(Ms, maxCol = 0.25, lmin = 1e-4, lmax = NULL)
write_out(ddcw$center, "ddcw_center")
write_out(ddcw$cov, "ddcw_cov")
ls0 <- estLocScale(M, center = FALSE, silent = TRUE)
Ms0 <- derived_input(scale(M, FALSE, ls0$scale), "ddcw_center0_input")
ddcw0 <- cellWise:::DDCWcov(Ms0, maxCol = 0.25, lmin = 1e-4, lmax = NULL, fixedCenter = TRUE)
write_out(ddcw0$cov, "ddcw_center0_cov")

run_cellmcd <- function(case, X, ...) {
  out <- cellMCD(X, checkPars = list(silent = TRUE, coreOnly = TRUE), ...)
  write_out(out$mu, paste0(case, "_location"))
  write_out(out$S, paste0(case, "_covariance"))
  write_out(out$raw.S, paste0(case, "_raw_covariance"))
  write_out(out$W, paste0(case, "_W"))
  write_out(out$preds, paste0(case, "_predictions"))
  write_out(out$csds, paste0(case, "_conditional_stds"))
  write_out(out$Zres, paste0(case, "_residuals"))
  write_out(data.frame(nosteps = out$nosteps), paste0(case, "_nosteps"))
}
run_cellmcd("cellmcd_default", M)
run_cellmcd("cellmcd_alpha05", M, alpha = 0.5, quant = 0.999)
Mc <- derived_input(sweep(M, 2, apply(M, 2, median, na.rm = TRUE)), "cellmcd_center0_input")
run_cellmcd("cellmcd_center0", Mc, fixedCenter = TRUE)
