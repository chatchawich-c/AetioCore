#!/usr/bin/env Rscript
# ติดตั้งแพ็กเกจครั้งเดียวก่อนรัน:
# install.packages(c("nanoparquet", "lavaan", "cSEM", "jsonlite", "future"))

args <- commandArgs(trailingOnly = TRUE)
engine   <- args[1]
pq_path  <- args[2]
fml_path <- args[3]
n_boot   <- as.integer(args[4])
n_runs   <- as.integer(args[5])
out_json <- args[6]

suppressPackageStartupMessages({
  library(nanoparquet)
  library(jsonlite)
  if (engine %in% c("semflow_r_cb", "semflow_r_pls", "semflow_r_plsc")) {
    library(semflow)
  } else if (engine == "lavaan_cb") {
    library(lavaan)
  } else if (engine %in% c("csem_pls", "csem_plsc", "csem_pls_par")) {
    library(cSEM)
  } else {
    stop(paste("Unknown R engine:", engine))
  }
})

formula_str <- paste(readLines(fml_path, warn = FALSE), collapse = "\n")

run_once <- function() {
  t0 <- proc.time()[["elapsed"]]
  df <- as.data.frame(nanoparquet::read_parquet(pq_path))
  t_load <- proc.time()[["elapsed"]]

  converged <- TRUE
  if (engine == "semflow_r_cb") {
    fit <- semflow::sem(formula_str, data = df, method = "cb", estimator = "ml", max_iter = 800L)
    converged <- isTRUE(fit$converged)
  } else if (engine == "semflow_r_pls") {
    fit <- semflow::sem(formula_str, data = df, method = "pls", n_boot = n_boot, max_iter = 300L, tol = 1e-6)
    converged <- isTRUE(fit$converged)
  } else if (engine == "semflow_r_plsc") {
    fit <- semflow::sem(formula_str, data = df, method = "plsc", n_boot = n_boot, max_iter = 300L, tol = 1e-6)
    converged <- isTRUE(fit$converged)
  } else if (engine == "lavaan_cb") {
    fit <- lavaan::sem(
      formula_str,
      data = df,
      likelihood = "wishart",
      information = "observed"
    )
    converged <- isTRUE(lavaan::lavInspect(fit, "converged"))
  } else if (engine %in% c("csem_pls", "csem_plsc", "csem_pls_par")) {
    disatt <- (engine == "csem_plsc")
    resamp <- if (n_boot > 0L) "bootstrap" else "none"
    plan   <- if (engine == "csem_pls_par") "multisession" else "sequential"
    fit <- cSEM::csem(
      .data = df,
      .model = formula_str,
      .PLS_weight_scheme_inner = "centroid",
      .disattenuate = disatt,
      .resample_method = resamp,
      .R = max(n_boot, 1L),
      .eval_plan = plan,
      .tolerance = 1e-6
    )
    converged <- TRUE
  }
  t_fit <- proc.time()[["elapsed"]]

  list(
    load_ms = (t_load - t0) * 1000.0,
    fit_ms = (t_fit - t_load) * 1000.0,
    total_ms = (t_fit - t0) * 1000.0,
    converged = converged
  )
}

# Warmup 1 รอบ (เว้นกรณี bootstrap 5,000 รอบใน cSEM เพื่อไม่ให้เสียเวลารอซ้ำ)
if (!(engine %in% c("csem_pls", "csem_plsc", "csem_pls_par") && n_boot >= 1000L)) {
  invisible(run_once())
}

loads <- numeric(n_runs)
fits  <- numeric(n_runs)
tots  <- numeric(n_runs)
convs <- logical(n_runs)

for (i in seq_len(n_runs)) {
  r <- run_once()
  loads[i] <- r$load_ms
  fits[i]  <- r$fit_ms
  tots[i]  <- r$total_ms
  convs[i] <- r$converged
}

res <- list(
  engine = engine,
  load_ms = round(median(loads), 2),
  fit_ms = round(median(fits), 2),
  total_ms = round(median(tots), 2),
  converged = all(convs)
)

write_json(res, out_json, auto_unbox = TRUE)
