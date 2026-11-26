parse_spec <- function(dsl_str) list(formula = dsl_str)
fit_sem <- function(spec, data, method = "cb", ...) list(spec = spec, method = method, n_obs = nrow(data))
fit_indices <- function(fit_result) fit_result$indices
propagate_risk <- function(fit_result, shocks) solve(diag(nrow(fit_result$B)) - fit_result$B, shocks)
