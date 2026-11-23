# Reference results of the authors' R code of cellPCA, cellRCov and cellMR for robpy's tests.
#
# The code accompanies Centofanti, Hubert & Rousseeuw (2026), "Cellwise and casewise robust
# multivariate regression with inference" (arXiv:2605.08002), and can be downloaded from
#   https://wis.kuleuven.be/statdatascience/code/cellmr_r_code.zip
# Run from the repository root, after generate_inputs.py, with the directory that contains
# cellMR_code.R, cellRCov_code.R, cellPCA_code.R and cpp_functions.cpp:
#
#   Rscript tests/reference/cellMR_reference.R path/to/cellmr_r_code
#
# Generated with R 4.5.1, cellWise 2.5.7, robustbase 0.99.6 and rrcov 1.7.7. The MCD of the
# scores (rrcov::CovMcd with nsamp = 2000) is random; its results are saved so that robpy can
# be checked with them.

args <- commandArgs(TRUE)
suppressMessages({library(cellWise); library(Rcpp); library(robustbase); library(rrcov)})
for (f in c("cellMR_code.R", "cellRCov_code.R", "cellPCA_code.R")) source(file.path(args[1], f))
Rcpp::sourceCpp(file.path(args[1], "cpp_functions.cpp"))

here <- "tests/reference"
out_dir <- file.path(here, "expected")
write_out <- function(x, name) {
  x <- as.data.frame(x)
  num <- vapply(x, is.numeric, logical(1))
  x[num] <- lapply(x[num], signif, digits = 10)
  write.csv(x, file.path(out_dir, paste0(name, ".csv")), row.names = FALSE)
}
write_mcd <- function(m, name) {
  write_out(m@center, paste0(name, "_center")); write_out(m@cov, paste0(name, "_cov"))
  write_out(m@raw.center, paste0(name, "_raw_center")); write_out(m@raw.cov, paste0(name, "_raw_cov"))
}

D <- as.matrix(read.csv(file.path(here, "data", "cell_mr.csv")))
Dnew <- as.matrix(read.csv(file.path(here, "data", "cell_mr_new.csv")))
X <- D[, 1:8]; Y <- D[, 9:11]

## cellPCA of the standardized joint data (as inside cellRCov), with MacroPCA start
scaleZ <- apply(D, 2, function(x) scale_tanh_new(na.omit(x) - median(x, na.rm = TRUE)))
Zs <- scale(D, center = FALSE, scale = scaleZ)
pars <- list(silent = TRUE, scale = FALSE, kmax = 3,
             DDCpars = list(tolProb = 0.99, fracNA = 0.9, cleanNAfirst = "columns"))
mp <- new_macropca(Zs, k = 2, MacroPCApars = pars)
write_out(scaleZ, "cellpca_scales")
write_out(mp$center, "macropca_center"); write_out(mp$loadings, "macropca_loadings")
write_out(mp$Fullimp$scoresfi, "macropca_scores")
pca <- cellPCA(as.matrix(Zs), k = 2, trace = FALSE, maxColFrac = 0.8, max_iter = 1000)
write_out(pca$U, "cellpca_U"); write_out(pca$V, "cellpca_V"); write_out(pca$mu, "cellpca_mu")
write_out(pca$eigenvalues, "cellpca_eigenvalues"); write_out(pca$weights_c, "cellpca_wc")
write_out(pca$weights_r, "cellpca_wr"); write_out(pca$Ximp, "cellpca_Ximp")
write_out(pca$rowres_std, "cellpca_rowres"); write_mcd(pca$mod_mcd, "cellpca_mcd")
Znew <- scale(Dnew, center = FALSE, scale = scaleZ)
new <- cellPCA_predict(Znew, pca, trace = FALSE)
write_out(new$U, "cellpca_new_U"); write_out(new$Ximp, "cellpca_new_Ximp")

## cellMR with imputation model for X
mr <- cellMR(X, Y, k = 2, lambda = 0.1, kX = 2, delta_reg = 0.5)
write_out(mr$B, "cellmr_B"); write_out(mr$b0, "cellmr_b0"); write_out(mr$mod_cellCov$S, "cellmr_S")
write_out(mr$mod_cellCov$mu, "cellmr_mu"); write_out(mr$S_eps_reg, "cellmr_S_eps")
write_out(mr$Ximp, "cellmr_Ximp"); write_out(mr$Yhat, "cellmr_Yhat")
write_out(mr$S_x_reg, "cellmr_S_x")
write_mcd(mr$mod_cellCov$mod_cellPCA$mod_mcd, "cellmr_mcd_joint")
write_mcd(mr$mod_cellPCA_X$mod_mcd, "cellmr_mcd_X")
pX <- cellPCA_predict(scale(Dnew[, 1:8], center = FALSE, scale = mr$scaleXfit * 0 + apply(X, 2,
  function(x) scale_tanh_new(na.omit(x) - median(x, na.rm = TRUE)))), mr$mod_cellPCA_X, trace = FALSE)
write_out(pX$Ximp, "cellmr_new_Ximp_std")
