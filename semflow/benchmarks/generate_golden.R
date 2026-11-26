#!/usr/bin/env Rscript
# ติดตั้งแพ็กเกจก่อนรันครั้งแรก:
# install.packages(c("lavaan", "cSEM", "jsonlite"))

suppressPackageStartupMessages({
  library(lavaan)
  library(cSEM)
  library(jsonlite)
})

out_dir <- file.path("benchmarks", "golden")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# -------------------------------------------------------------------------
# 1. ส่งออกชุดข้อมูลมาตรฐานเป็น CSV (ให้ Python อ่านชุดข้อมูลเดียวกันเป๊ะ)
# -------------------------------------------------------------------------
data("HolzingerSwineford1939", package = "lavaan")
data("PoliticalDemocracy", package = "lavaan")

hs_df <- HolzingerSwineford1939[, paste0("x", 1:9)]
pd_df <- PoliticalDemocracy[, c(paste0("y", 1:8), paste0("x", 1:3))]

write.csv(hs_df, file.path(out_dir, "HolzingerSwineford1939.csv"), row.names = FALSE)
write.csv(pd_df, file.path(out_dir, "PoliticalDemocracy.csv"), row.names = FALSE)

# ฟังก์ชันช่วยแปลงผลลัพธ์ lavaan เป็น list สำหรับ JSON
extract_lavaan <- function(fit) {
  pe <- parameterEstimates(fit)
  pe$se[is.na(pe$se)] <- 0.0
  fm <- fitMeasures(
    fit,
    c("chisq", "df", "cfi", "tli", "rmsea", "srmr_bentler_nomean", "fmin")
  )
  list(
    nobs = lavInspect(fit, "nobs"),
    fit = list(
      chi_square = unname(fm["chisq"]),
      df = as.integer(unname(fm["df"])),
      cfi = unname(fm["cfi"]),
      tli = unname(fm["tli"]),
      rmsea = unname(fm["rmsea"]),
      srmr = unname(fm["srmr_bentler_nomean"]),
      f_ml = unname(fm["fmin"]) * 2.0
    ),
    params = lapply(seq_len(nrow(pe)), function(i) {
      list(
        lhs = as.character(pe$lhs[i]),
        op  = as.character(pe$op[i]),
        rhs = as.character(pe$rhs[i]),
        est = as.numeric(pe$est[i]),
        se  = as.numeric(pe$se[i])
      )
    })
  )
}

# -------------------------------------------------------------------------
# 2. CB-SEM Golden Fixture #1: HolzingerSwineford1939 (CFA 3 ปัจจัย)
# -------------------------------------------------------------------------
hs_model <- "
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
"

fit_hs <- cfa(
  hs_model,
  data = hs_df,
  likelihood = "wishart",
  information = "observed"
)

write_json(
  extract_lavaan(fit_hs),
  file.path(out_dir, "lavaan_hs1939.json"),
  auto_unbox = TRUE,
  pretty = TRUE,
  digits = 10
)

# -------------------------------------------------------------------------
# 3. CB-SEM Golden Fixture #2: PoliticalDemocracy (Full SEM + Residual Cov)
# -------------------------------------------------------------------------
pd_cb_model <- "
  ind60 =~ x1 + x2 + x3
  dem60 =~ y1 + y2 + y3 + y4
  dem65 =~ y5 + y6 + y7 + y8
  dem60 ~ ind60
  dem65 ~ ind60 + dem60
  y1 ~~ y5
  y2 ~~ y4 + y6
  y3 ~~ y7
  y4 ~~ y8
  y6 ~~ y8
"

fit_pd <- sem(
  pd_cb_model,
  data = pd_df,
  likelihood = "wishart",
  information = "observed"
)

write_json(
  extract_lavaan(fit_pd),
  file.path(out_dir, "lavaan_poldem.json"),
  auto_unbox = TRUE,
  pretty = TRUE,
  digits = 10
)

# -------------------------------------------------------------------------
# 4. PLS-SEM & PLSc Golden Fixtures: cSEM บน PoliticalDemocracy (Structural)
# -------------------------------------------------------------------------
pd_pls_model <- "
  ind60 =~ x1 + x2 + x3
  dem60 =~ y1 + y2 + y3 + y4
  dem65 =~ y5 + y6 + y7 + y8
  dem60 ~ ind60
  dem65 ~ ind60 + dem60
"

extract_csem <- function(res) {
  pmat <- res$Estimates$Path_estimates
  lmat <- res$Estimates$Loading_estimates
  q    <- assess(res, .quality_criterion = c("ave", "rho_c", "r2"))

  paths <- list()
  for (r in rownames(pmat)) {
    for (c in colnames(pmat)) {
      if (abs(pmat[r, c]) > 1e-12) {
        paths[[length(paths) + 1]] <- list(from = c, to = r, est = as.numeric(pmat[r, c]))
      }
    }
  }

  loadings <- list()
  for (r in rownames(lmat)) {
    for (c in colnames(lmat)) {
      if (abs(lmat[r, c]) > 1e-12) {
        loadings[[length(loadings) + 1]] <- list(latent = r, indicator = c, est = as.numeric(lmat[r, c]))
      }
    }
  }

  list(
    paths = paths,
    loadings = loadings,
    rho_a = as.list(res$Estimates$Reliabilities),
    ave   = as.list(q$AVE),
    cr    = as.list(calculateRhoC(res)),
    r2    = as.list(q$R2)
  )
}

fit_pls <- csem(
  .data = pd_df,
  .model = pd_pls_model,
  .PLS_weight_scheme_inner = "centroid",
  .disattenuate = FALSE,
  .tolerance = 1e-08
)

fit_plsc <- csem(
  .data = pd_df,
  .model = pd_pls_model,
  .PLS_weight_scheme_inner = "centroid",
  .disattenuate = TRUE,
  .tolerance = 1e-08
)

write_json(
  extract_csem(fit_pls),
  file.path(out_dir, "csem_poldem_pls.json"),
  auto_unbox = TRUE,
  pretty = TRUE,
  digits = 10
)

write_json(
  extract_csem(fit_plsc),
  file.path(out_dir, "csem_poldem_plsc.json"),
  auto_unbox = TRUE,
  pretty = TRUE,
  digits = 10
)

cat("Generated all golden fixtures in:", out_dir, "\n")
