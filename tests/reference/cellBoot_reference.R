# Reference results of the authors' R code of cellBoot for robpy's tests.
#
# As cellMR_reference.R (same code, from
#   https://wis.kuleuven.be/statdatascience/code/cellmr_r_code.zip):
#
#   Rscript tests/reference/cellBoot_reference.R path/to/cellmr_r_code
#
# The first part replays the steps of cellBoot up to the weighted estimate of the observed
# data (deterministic given the MCD of the cellPCA scores, which is saved). Then cellBoot
# itself runs, whose bias correction and bootstrap are random; robpy is compared with it
# within a tolerance. cellBoot needs ncores > 1 (with ncores = 1 it refers to an undefined
# variable). Generated with R 4.5.1, cellWise 2.5.7, robustbase 0.99.6 and rrcov 1.7.7.

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

D <- as.matrix(read.csv(file.path(here, "data", "cell_mr.csv")))
X <- D[, 1:8]; Y <- D[, 9:11]
k <- 2; p <- ncol(X); q <- ncol(Y); n <- nrow(X); Z <- cbind(X, Y)

o <- cellRCov(Z, k = k, kmax = min(30, p + q - 1), reg = T, delta_reg = 0, ncores = 1, scale = T,
              control.cellPCA = list(max_iter = 1000, trace = F, V_est = T))
pca <- o$mod_cellPCA; scaleX <- o$scaleX; Zs <- col_div_arma(Z, scaleX)
pca$par <- get_tuning_const_rho2_fast(p + q)$par
par2 <- c(1.5, 4, unlist(calculateq1q2(b = 1.5, c = 4)))
cut1 <- qchisq(0.99, df = k); cut2 <- qchisq(0.9999, df = k)
par_mcd <- c(cut1, cut2, unlist(calculateq1q2(b = cut1, c = cut2)))
DDC <- pca$mod_macropca$DDC; DDC$locX <- o$mustd; DDC$scaleX <- diag(o$Sstd)
cc <- DDCpredict2weight_compute_const(Zs, InitialDDC = DDC)
DDC$deshrinkage <- cc$deshrinkage; DDC$scalestres <- cc$scalestres
L <- chol_cache2(pca$mod_mcd@raw.cov)
pred <- DDCpredict2weight_cpp(Zs, DDC)
w0 <- get_weights_sub(Zs, pred$Ximp, pca, M = pca$M, scale = T, par2 = par2)
w <- get_weights_sub(Zs, pred$Ximp, pca, M = pca$M, scale = c(w0$scale_1, w0$scale_2), par2 = par2)
ind_r <- which(w$weights_r == 0); ws <- rep(0, n)
if (length(ind_r) > 0) {
  ws[-ind_r] <- get_weights_mcd_tanh(as.matrix(w$U[-ind_r, ]), pca$mod_mcd@raw.center, L, par_mcd)
} else {
  ws <- get_weights_mcd_tanh(w$U, pca$mod_mcd@raw.center, L, par_mcd)
}
S_err <- get_S_err_cpp3(col_center_arma(Zs, pca$mu), row_scale_arma(w$W, sqrt(ws)))
S_B <- diag(scaleX) %*% cov_ridge(S_err, o$delta_reg) %*% diag(scaleX)
Wmu <- row_scale_arma(w$weights_c, w$weights_r * ws)
mu_B <- diag(scaleX) %*% (colSums(Wmu * Zs, na.rm = T) / colSums(Wmu))
write_out(pred$Ximp, "cellboot_Ximp"); write_out(ws, "cellboot_weights_sub")
write_out(w$weights_r, "cellboot_weights_r"); write_out(S_B, "cellboot_S_obs"); write_out(mu_B, "cellboot_mu_obs")
m <- pca$mod_mcd
write_out(m@center, "cellboot_mcd_center"); write_out(m@cov, "cellboot_mcd_cov")
write_out(m@raw.center, "cellboot_mcd_raw_center"); write_out(m@raw.cov, "cellboot_mcd_raw_cov")

res <- cellBoot(X, Y, k = k, lambda = 0.1, lin_comb = diag(p * q + q), ncores = 2, B = 200)
write_out(c(res$b_unb_obs), "cellboot_intercepts"); write_out(res$B_unb_obs, "cellboot_slopes")
write_out(res$CI, "cellboot_intervals")
