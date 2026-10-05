## ============================================================
## ACTL3301/5301 Models for Risk Management -- Assignment 1
## Term 2 2026, UNSW Sydney
## Author: Corinne C.
##
## Copula-based tail risk and time series analysis of NVDA and AMD
## ============================================================

## ------------------------------------------------------------
## Packages (installed only if missing)
## ------------------------------------------------------------
pkgs <- c("quantmod", "ggplot2", "tseries", "forecast", "VineCopula")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
}

library(quantmod)
library(ggplot2)

## ------------------------------------------------------------
## 0. DATA PREPARATION
## ------------------------------------------------------------

start_date <- as.Date("2016-06-01")
end_date   <- as.Date("2026-06-06")   # not inclusive
tickers    <- c("NVDA", "AMD")

getSymbols(tickers, src = "yahoo", from = start_date, to = end_date, auto.assign = TRUE)

Adj_NVDA <- Ad(NVDA)
Adj_AMD  <- Ad(AMD)

X1 <- (log(Adj_NVDA) - log(lag(Adj_NVDA, 1)))[-1]
X1 <- as.numeric(coredata(X1))
X2 <- (log(Adj_AMD) - log(lag(Adj_AMD, 1)))[-1]
X2 <- as.numeric(coredata(X2))

ret_dates <- index(Adj_NVDA)[-1]

n_obs <- length(X1)
stopifnot(length(X1) == length(X2))

# Losses per $1 invested
L1 <- 1 - exp(X1)
L2 <- 1 - exp(X2)

## Global settings required by the brief
n     <- 10000
alpha <- c(0.95, 0.99, 0.995)

reset_rng <- function(seed = 2026) {
  RNGversion("4.3.0")
  set.seed(seed)
}

## ------------------------------------------------------------
## Generic empirical VaR / ES helpers (used throughout)
## ------------------------------------------------------------
## VaR_alpha(L): alpha-quantile of the loss distribution.
## ES_alpha(L):  average loss beyond (and including) that quantile.
##
## type = 1 (the quantile() default step-function / inverse-ecdf quantile)
## is used deliberately throughout: it is the *same* quantile definition
## implied by inverting ecdf(), which is what the brief requires us to use
## for the marginals ("use the empirical cumulative distribution function
## (via ecdf) of the marginal distributions"). Keeping VaR/ES estimation
## and copula-marginal inversion on the same quantile convention avoids a
## silent mismatch between the two.

VaR_emp <- function(L, alpha, type = 1) {
  as.numeric(quantile(L, probs = alpha, type = type))
}

ES_emp <- function(L, alpha, type = 1) {
  sapply(alpha, function(a) {
    v <- as.numeric(quantile(L, probs = a, type = type))
    mean(L[L >= v])
  })
}

risk_table <- function(L, alpha, label) {
  data.frame(
    portfolio = label,
    alpha     = alpha,
    VaR       = VaR_emp(L, alpha),
    ES        = ES_emp(L, alpha)
  )
}

## ============================================================
## TASK 1: VaR / ES of L1 (NVDA) and L2 (AMD)
## ============================================================

tbl_L1 <- risk_table(L1, alpha, "NVDA (L1)")
tbl_L2 <- risk_table(L2, alpha, "AMD (L2)")
task1_results <- rbind(tbl_L1, tbl_L2)
print(task1_results)

## ============================================================
## TASK 2: Kendall's tau (direct concordance/discordance count)
##          and Spearman's rho (empirical distribution based)
## ============================================================

kendall_tau_direct <- function(x, y) {
  n_ <- length(x)
  dx <- outer(x, x, "-")
  dy <- outer(y, y, "-")
  s  <- sign(dx * dy)
  s_upper <- s[upper.tri(s)]
  n_conc <- sum(s_upper > 0)
  n_disc <- sum(s_upper < 0)
  n_tied <- sum(s_upper == 0)
  total_pairs <- choose(n_, 2)
  stopifnot(n_conc + n_disc + n_tied == total_pairs)
  
  tau <- (n_conc - n_disc) / total_pairs  # formal tau-a definition
  list(tau = tau, n_conc = n_conc, n_disc = n_disc, n_tied = n_tied,
       total_pairs = total_pairs)
}

kendall_res <- kendall_tau_direct(L1, L2)
cat("Kendall's tau (direct count, formal C(n,2) denominator):",
    kendall_res$tau, "\n")
cat("Tied pairs observed:", kendall_res$n_tied, "out of",
    kendall_res$total_pairs, "total pairs\n")

u1 <- rank(L1) / (length(L1) + 1)
u2 <- rank(L2) / (length(L2) + 1)
spearman_rho_emp <- cor(u1, u2, method = "pearson")
cat("Spearman's rho (empirical):", spearman_rho_emp, "\n")
cat("Range check -- min/max pseudo-observations:",
    range(c(u1, u2)), "(should be strictly inside (0,1))\n")

## Sanity check against built-in estimators (report only, not the "answer"):
cat("cor(L1,L2, method='kendall') =", cor(L1, L2, method = "kendall"), "\n")
cat("cor(L1,L2, method='spearman')=", cor(L1, L2, method = "spearman"), "\n")

## Equally weighted portfolio loss (used from here on)
L_port_empirical <- 0.5 * (L1 + L2)

## ============================================================
## TASK 3: Independence copula
## ============================================================

reset_rng()
U1_ind <- runif(n)
U2_ind <- runif(n)

L1_sim_ind <- VaR_emp(L1, U1_ind)
L2_sim_ind <- VaR_emp(L2, U2_ind)
L_sim_ind  <- 0.5 * (L1_sim_ind + L2_sim_ind)

VaR_ind <- VaR_emp(L_sim_ind, alpha)
ES_ind  <- ES_emp(L_sim_ind, alpha)
task3_results <- data.frame(alpha = alpha, VaR = VaR_ind, ES = ES_ind)
print(task3_results)

ggplot(data.frame(U1 = U1_ind, U2 = U2_ind), aes(U1, U2)) +
  geom_point(alpha = 0.15, size = 0.6) +
  labs(title = "Simulated samples: Independence copula") +
  theme_minimal()

## ============================================================
## TASK 4: Gaussian copula
## ============================================================

rho_hat <- cor(X1, X2)
cat("rho_hat (Gaussian copula) =", rho_hat, "\n")

simulate_bvn <- function(n, rho) {
  Z1 <- rnorm(n)
  Z2 <- rnorm(n)
  Y1 <- Z1
  Y2 <- rho * Z1 + sqrt(1 - rho^2) * Z2
  cbind(Y1, Y2)
}

reset_rng()
Z_gauss <- simulate_bvn(n, rho_hat)
U_gauss <- pnorm(Z_gauss)

L1_sim_gauss <- VaR_emp(L1, U_gauss[, 1])
L2_sim_gauss <- VaR_emp(L2, U_gauss[, 2])
L_sim_gauss  <- 0.5 * (L1_sim_gauss + L2_sim_gauss)

VaR_gauss <- VaR_emp(L_sim_gauss, alpha)
ES_gauss  <- ES_emp(L_sim_gauss, alpha)
task4_results <- data.frame(alpha = alpha, VaR = VaR_gauss, ES = ES_gauss)
print(task4_results)

ggplot(data.frame(U1 = U_gauss[, 1], U2 = U_gauss[, 2]), aes(U1, U2)) +
  geom_point(alpha = 0.15, size = 0.6) +
  labs(title = paste0("Simulated samples: Gaussian copula (rho = ",
                      round(rho_hat, 3), ")")) +
  theme_minimal()

## ============================================================
## TASK 5: t-copulae, nu = 3, 10, 10000
## ============================================================

simulate_bvt <- function(n, rho, nu) {
  Z <- simulate_bvn(n, rho)
  W <- rchisq(n, df = nu)
  T <- Z / sqrt(W / nu)
  T
}

nu_vec <- c(3, 10, 10000)
t_copula_results <- list()
t_copula_samples <- list()

for (nu in nu_vec) {
  reset_rng()
  Tsim  <- simulate_bvt(n, rho_hat, nu)
  Usim  <- pt(Tsim, df = nu)
  
  L1_sim <- VaR_emp(L1, Usim[, 1])
  L2_sim <- VaR_emp(L2, Usim[, 2])
  L_sim  <- 0.5 * (L1_sim + L2_sim)
  
  t_copula_results[[as.character(nu)]] <- data.frame(
    nu = nu, alpha = alpha,
    VaR = VaR_emp(L_sim, alpha),
    ES  = ES_emp(L_sim, alpha)
  )
  t_copula_samples[[as.character(nu)]] <- Usim
}

task5_results <- do.call(rbind, t_copula_results)
print(task5_results)

for (nu in nu_vec) {
  Usim <- t_copula_samples[[as.character(nu)]]
  p <- ggplot(data.frame(U1 = Usim[, 1], U2 = Usim[, 2]), aes(U1, U2)) +
    geom_point(alpha = 0.15, size = 0.6) +
    labs(title = paste0("Simulated samples: t-copula (nu = ", nu, ")")) +
    theme_minimal()
  print(p)
}

## Comment: as nu -> infinity the t-copula -> Gaussian copula
## (compare nu = 10000 plot to Task 4's plot). Small nu (e.g. 3) produces
## visibly denser clustering of points in BOTH corners simultaneously --
## symmetric tail dependence the Gaussian copula lacks asymptotically.

## ============================================================
## TASK 6: Comonotonicity and countermonotonicity copulae
## ============================================================

reset_rng()
U_comon <- runif(n)
U1_comon <- U_comon
U2_comon <- U_comon

reset_rng()
U_counter <- runif(n)
U1_counter <- U_counter
U2_counter <- 1 - U_counter

L1_sim_comon <- VaR_emp(L1, U1_comon)
L2_sim_comon <- VaR_emp(L2, U2_comon)
L_sim_comon  <- 0.5 * (L1_sim_comon + L2_sim_comon)

L1_sim_counter <- VaR_emp(L1, U1_counter)
L2_sim_counter <- VaR_emp(L2, U2_counter)
L_sim_counter  <- 0.5 * (L1_sim_counter + L2_sim_counter)

task6_results <- rbind(
  data.frame(copula = "Comonotonicity",     alpha = alpha,
             VaR = VaR_emp(L_sim_comon, alpha),   ES = ES_emp(L_sim_comon, alpha)),
  data.frame(copula = "Countermonotonicity", alpha = alpha,
             VaR = VaR_emp(L_sim_counter, alpha), ES = ES_emp(L_sim_counter, alpha))
)
print(task6_results)

## Purpose: Frechet-Hoeffding upper/lower bounds -- worst-case
## (comonotone, no diversification) and best-case (countermonotone, maximal
## hedging) benchmarks for the given fixed marginals.

## ============================================================
## TASK 7: Compare all models vs the empirical benchmark
## ============================================================

task7_empirical <- data.frame(copula = "Empirical", alpha = alpha,
                              VaR = VaR_emp(L_port_empirical, alpha),
                              ES  = ES_emp(L_port_empirical, alpha))

task7_all <- rbind(
  task7_empirical,
  data.frame(copula = "Independence", alpha = alpha, VaR = VaR_ind,  ES = ES_ind),
  data.frame(copula = "Gaussian",     alpha = alpha, VaR = VaR_gauss, ES = ES_gauss),
  data.frame(copula = paste0("t (nu=", task5_results$nu, ")"),
             alpha = task5_results$alpha, VaR = task5_results$VaR, ES = task5_results$ES),
  data.frame(copula = "Comonotonicity",      alpha = alpha,
             VaR = VaR_emp(L_sim_comon, alpha),   ES = ES_emp(L_sim_comon, alpha)),
  data.frame(copula = "Countermonotonicity", alpha = alpha,
             VaR = VaR_emp(L_sim_counter, alpha), ES = ES_emp(L_sim_counter, alpha))
)
print(task7_all)


cond_crash_prob <- function(U1, U2, alpha) {
  # U1, U2 are pseudo-observations on the LOSS scale.
  # Event of interest: {U_L1 >= alpha, U_L2 >= alpha}  (joint large losses,
  # i.e. F1(X1) <= 1-alpha and F2(X2) <= 1-alpha on the return scale).
  sapply(alpha, function(a) {
    joint <- mean(U1 >= a & U2 >= a)
    joint / (1 - a)
  })
}

cond_probs <- data.frame(
  alpha              = alpha,
  Empirical          = cond_crash_prob(u1, u2, alpha),
  Independence       = cond_crash_prob(U1_ind, U2_ind, alpha),
  Gaussian           = cond_crash_prob(U_gauss[, 1], U_gauss[, 2], alpha),
  t_nu3              = cond_crash_prob(t_copula_samples[["3"]][, 1],
                                       t_copula_samples[["3"]][, 2], alpha),
  t_nu10             = cond_crash_prob(t_copula_samples[["10"]][, 1],
                                       t_copula_samples[["10"]][, 2], alpha),
  t_nu10000          = cond_crash_prob(t_copula_samples[["10000"]][, 1],
                                       t_copula_samples[["10000"]][, 2], alpha),
  Comonotonicity     = cond_crash_prob(U1_comon, U2_comon, alpha),
  Countermonotonicity = cond_crash_prob(U1_counter, U2_counter, alpha)
)
print(cond_probs)

## ============================================================
## TASK 8: BiCopSelect (VineCopula) -- best-fitting ARCHIMEDEAN copula
## ============================================================
## VineCopula is explicitly allowed/required for this task only.

library(VineCopula)

archimedean_families <- c(3, 4, 5, 6, 7, 8, 9, 10)
# 3 = Clayton, 4 = Gumbel, 5 = Frank, 6 = Joe,
# 7 = BB1, 8 = BB6, 9 = BB7, 10 = BB8

best_cop <- BiCopSelect(u1, u2, familyset = archimedean_families,
                        selectioncrit = "AIC")
summary(best_cop)
print(best_cop)

cat("Selected family code:", best_cop$family,
    " | family name:", BiCopName(best_cop$family, short = FALSE), "\n")
cat("Parameter(s): par =", best_cop$par, " par2 =", best_cop$par2, "\n")
cat("Kendall's tau implied by fit:", best_cop$tau, "\n")
cat("AIC:", best_cop$AIC, " BIC:", best_cop$BIC, "\n")
cat("Upper tail dependence lambda_U:", BiCopPar2TailDep(best_cop)$upper, "\n")
cat("Lower tail dependence lambda_L:", BiCopPar2TailDep(best_cop)$lower, "\n")

## Simulate from the selected copula and repeat the VaR/ES exercise for
## direct comparison with Task 7's table.
reset_rng()
U_best <- BiCopSim(n, best_cop)
L1_sim_best <- VaR_emp(L1, U_best[, 1])
L2_sim_best <- VaR_emp(L2, U_best[, 2])
L_sim_best  <- 0.5 * (L1_sim_best + L2_sim_best)

task8_results <- data.frame(alpha = alpha,
                            VaR = VaR_emp(L_sim_best, alpha),
                            ES  = ES_emp(L_sim_best, alpha))
print(task8_results)

task8_crash_prob <- cond_crash_prob(U_best[, 1], U_best[, 2], alpha)
print(data.frame(alpha = alpha, crash_prob = task8_crash_prob))


## ============================================================
## ============================================================
##                    TIME SERIES TASKS
## ============================================================
## ============================================================

library(tseries)
library(forecast)

## ------------------------------------------------------------
## TS TASK 1: Plots and SACF
## ------------------------------------------------------------

prices_df <- data.frame(date = ret_dates,
                        NVDA = as.numeric(Adj_NVDA)[-1],
                        AMD  = as.numeric(Adj_AMD)[-1])
returns_df <- data.frame(date = ret_dates, X1 = X1, X2 = X2)

ggplot(prices_df, aes(date, NVDA)) + geom_line() +
  labs(title = "NVDA adjusted close price", y = "Price (USD)") + theme_minimal()
ggplot(prices_df, aes(date, AMD)) + geom_line() +
  labs(title = "AMD adjusted close price", y = "Price (USD)") + theme_minimal()
ggplot(returns_df, aes(date, X1)) + geom_line() +
  labs(title = "NVDA daily log-return", y = "Return") + theme_minimal()
ggplot(returns_df, aes(date, X2)) + geom_line() +
  labs(title = "AMD daily log-return", y = "Return") + theme_minimal()

## Writing to a PDF with an explicit size avoids the "figure margins too
## large" error some graphics devices throw on a 2x2 layout.
pdf("acf_plots.pdf", width = 10, height = 8)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
acf(as.numeric(Adj_NVDA), lag.max = 15, main = "SACF: NVDA price")
acf(as.numeric(Adj_AMD),  lag.max = 15, main = "SACF: AMD price")
acf(X1, lag.max = 15, main = "SACF: NVDA log-return")
acf(X2, lag.max = 15, main = "SACF: AMD log-return")
par(mfrow = c(1, 1))
dev.off()

## Formal stationarity check (ADF test) to support the visual call
adf.test(as.numeric(Adj_NVDA))
adf.test(as.numeric(Adj_AMD))
adf.test(X1)
adf.test(X2)

## ------------------------------------------------------------
## TS TASK 2: MA(3) parameters from SACF via method of moments
## ------------------------------------------------------------

fit_MA3_moments <- function(x) {
  mu_hat   <- mean(x)
  acf_obj  <- acf(x, lag.max = 3, plot = FALSE)
  rho1 <- acf_obj$acf[2]
  rho2 <- acf_obj$acf[3]
  rho3 <- acf_obj$acf[4]
  gamma0_hat <- var(x) * (length(x) - 1) / length(x)
  
  eqs <- function(th) {
    th1 <- th[1]; th2 <- th[2]; th3 <- th[3]
    S <- 1 + th1^2 + th2^2 + th3^2
    e1 <- rho1 * S - (th1 + th1 * th2 + th2 * th3)
    e2 <- rho2 * S - (th2 + th1 * th3)
    e3 <- rho3 * S - th3
    c(e1, e2, e3)
  }
  obj <- function(th) sum(eqs(th)^2)
  
  starts <- expand.grid(th1 = c(-0.5, 0, 0.5),
                        th2 = c(-0.5, 0, 0.5),
                        th3 = c(-0.5, 0, 0.5))
  best <- NULL
  for (i in seq_len(nrow(starts))) {
    fit <- try(optim(as.numeric(starts[i, ]), obj, method = "BFGS",
                     control = list(maxit = 2000)), silent = TRUE)
    if (inherits(fit, "try-error")) next
    th <- fit$par
    roots <- polyroot(c(1, th[1], th[2], th[3]))
    invertible <- all(Mod(roots) > 1)
    if (invertible && fit$value < 1e-8) {
      if (is.null(best) || fit$value < best$value) best <- fit
    }
  }
  if (is.null(best)) stop("No invertible root found -- widen the starting grid.")
  
  th_hat <- best$par
  S_hat  <- 1 + sum(th_hat^2)
  sig2_hat <- gamma0_hat / S_hat
  
  list(mu = mu_hat, theta = th_hat, sigma2 = sig2_hat,
       rho_used = c(rho1, rho2, rho3), residual = best$value)
}

fit1 <- fit_MA3_moments(X1)   # NVDA
fit2 <- fit_MA3_moments(X2)   # AMD

cat("NVDA MA(3) moment estimates:\n"); print(fit1)
cat("AMD  MA(3) moment estimates:\n"); print(fit2)

## Report: express e.g. X1,t = mu1_hat + Z_t + th1_1*Z_{t-1} + th1_2*Z_{t-2}
## + th1_3*Z_{t-3}, Z_t ~ WN(0, sigma1_hat^2), substituting the fitted
## numeric values (and similarly for AMD).

## ------------------------------------------------------------
## TS TASK 3: auto.arima
## ------------------------------------------------------------

fit1_auto <- auto.arima(X1, ic = "aic")
fit2_auto <- auto.arima(X2, ic = "aic")
summary(fit1_auto)
summary(fit2_auto)


## ------------------------------------------------------------
## TS TASK 4: 5-day-ahead forecast via Box-Jenkins (manual recursion,
##            NOT `predict`)
## ------------------------------------------------------------

fc_start <- end_date
fc_end   <- as.Date(end_date) + 20   # generous window; trimmed to 5 obs below
getSymbols(tickers, src = "yahoo", from = fc_start, to = fc_end, auto.assign = TRUE)
Adj_NVDA_fut <- Ad(NVDA)[1:5]
Adj_AMD_fut  <- Ad(AMD)[1:5]

## True next-5-day returns need the last in-sample price too, so re-pull a
## small overlapping window:
getSymbols(tickers, src = "yahoo",
           from = as.Date(end_date) - 3, to = fc_end, auto.assign = TRUE)
Adj_NVDA_ext <- Ad(NVDA)
Adj_AMD_ext  <- Ad(AMD)
X1_future <- as.numeric(coredata((log(Adj_NVDA_ext) - log(lag(Adj_NVDA_ext, 1)))[-1]))[1:5]
X2_future <- as.numeric(coredata((log(Adj_AMD_ext)  - log(lag(Adj_AMD_ext, 1)))[-1]))[1:5]

## Manual Box-Jenkins forecast recursion for a fitted ARMA(p,q) model from
## auto.arima. Uses the model's own coefficients and in-sample residuals
## (via resid()) as the known past innovations, then forecasts forward
## assuming E[Z_{T+h}] = 0 for h > 0.
forecast_manual <- function(fit, h = 5) {
  cf   <- coef(fit)
  ar_names <- grep("^ar", names(cf), value = TRUE)
  ma_names <- grep("^ma", names(cf), value = TRUE)
  phi  <- if (length(ar_names)) cf[ar_names] else numeric(0)
  theta <- if (length(ma_names)) cf[ma_names] else numeric(0)
  mu   <- if ("intercept" %in% names(cf)) cf["intercept"] else 0
  
  p <- length(phi); q <- length(theta)
  x <- as.numeric(fit$x)          # original series used to fit
  z <- as.numeric(resid(fit))     # in-sample innovations (Box-Jenkins residuals)
  Tn <- length(x)
  
  x_ext <- c(x, rep(NA, h))
  z_ext <- c(z, rep(0, h))        # future innovations set to their expectation, 0
  
  for (k in 1:h) {
    t <- Tn + k
    ar_part <- 0
    if (p > 0) {
      for (i in 1:p) {
        ar_part <- ar_part + phi[i] * (x_ext[t - i] - mu)
      }
    }
    ma_part <- 0
    if (q > 0) {
      for (j in 1:q) {
        ma_part <- ma_part + theta[j] * z_ext[t - j]
      }
    }
    x_ext[t] <- mu + ar_part + ma_part
  }
  x_ext[(Tn + 1):(Tn + h)]
}

fc1 <- forecast_manual(fit1_auto, h = 5)
fc2 <- forecast_manual(fit2_auto, h = 5)

comparison_df <- data.frame(
  day        = 1:5,
  NVDA_pred  = fc1, NVDA_actual = X1_future,
  AMD_pred   = fc2, AMD_actual  = X2_future
)
print(comparison_df)

rmse <- function(pred, actual) sqrt(mean((pred - actual)^2))
mae  <- function(pred, actual) mean(abs(pred - actual))
cat("NVDA forecast RMSE:", rmse(fc1, X1_future), " MAE:", mae(fc1, X1_future), "\n")
cat("AMD  forecast RMSE:", rmse(fc2, X2_future), " MAE:", mae(fc2, X2_future), "\n")


## ------------------------------------------------------------
## TS TASK 5: Residual diagnostics -- should behave like white noise
## ------------------------------------------------------------

res1 <- resid(fit1_auto)
res2 <- resid(fit2_auto)

lb1 <- Box.test(res1, lag = 15, type = "Ljung-Box", fitdf = length(coef(fit1_auto)))
lb2 <- Box.test(res2, lag = 15, type = "Ljung-Box", fitdf = length(coef(fit2_auto)))
print(lb1)
print(lb2)

par(mfrow = c(1, 2))
acf(res1, lag.max = 15, main = "Residual ACF: NVDA model")
acf(res2, lag.max = 15, main = "Residual ACF: AMD model")
par(mfrow = c(1, 1))
