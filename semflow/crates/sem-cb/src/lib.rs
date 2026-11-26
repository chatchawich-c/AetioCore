use faer::prelude::*;
use faer::Side;
use rayon::prelude::*;
use sem_autodiff::{minimize_lbfgs, Tape};
use sem_dsl::{CausalGraphIR, ParamSpec};
use sem_traits::{MatrixView, SemFloat};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct CbParamEstimate {
    pub lhs: String,
    pub op: String,
    pub rhs: String,
    pub estimate: f64,
    pub se: f64,
    pub fixed: bool,
}

#[derive(Debug, Clone)]
pub struct FitIndices {
    pub chi_square: f64,
    pub df: usize,
    pub cfi: f64,
    pub tli: f64,
    pub rmsea: f64,
    pub srmr: f64,
}

#[derive(Debug, Clone)]
pub struct CbResult {
    pub converged: bool,
    pub iterations: usize,
    pub f_ml: f64,
    pub params: Vec<CbParamEstimate>,
    pub fit: FitIndices,
}

struct MatrixSlot {
    row: usize,
    col: usize,
    is_symmetric_s: bool, // true = S_ram, false = A
    is_variance: bool,    // true = diagonal variance (ใช้ softplus)
    param_idx: Option<usize>,
    fixed_val: f64,
    lhs: String,
    op: String,
    rhs: String,
}

/// คำนวณ Sample Covariance Matrix S (p x p) แบบขนานด้วย rayon (Zero-Copy)
fn sample_covariance<T: SemFloat>(data: MatrixView<'_, T>, cols: &[usize]) -> Mat<f64> {
    let n = data.nrows();
    let p = cols.len();
    let chunk_size = 4096.max(n / (rayon::current_num_threads() * 4).max(1));
    let n_chunks = (n + chunk_size - 1) / chunk_size;

    let sums = (0..n_chunks)
        .into_par_iter()
        .map(|c| {
            let start = c * chunk_size;
            let end = (start + chunk_size).min(n);
            let mut local = vec![0.0_f64; p];
            for i in start..end {
                for j in 0..p {
                    local[j] += data.get(i, cols[j]);
                }
            }
            local
        })
        .reduce(
            || vec![0.0_f64; p],
            |mut a, b| {
                for j in 0..p {
                    a[j] += b[j];
                }
                a
            },
        );

    let means: Vec<f64> = sums.into_iter().map(|s| s / (n as f64)).collect();

    let tri_len = p * (p + 1) / 2;
    let tri_sums = (0..n_chunks)
        .into_par_iter()
        .map(|c| {
            let start = c * chunk_size;
            let end = (start + chunk_size).min(n);
            let mut local = vec![0.0_f64; tri_len];
            let mut row_d = vec![0.0_f64; p];
            for i in start..end {
                for a in 0..p {
                    row_d[a] = data.get(i, cols[a]) - means[a];
                }
                let mut idx = 0;
                for a in 0..p {
                    let da = row_d[a];
                    for b in a..p {
                        local[idx] += da * row_d[b];
                        idx += 1;
                    }
                }
            }
            local
        })
        .reduce(
            || vec![0.0_f64; tri_len],
            |mut a, b| {
                for k in 0..tri_len {
                    a[k] += b[k];
                }
                a
            },
        );

    let mut cov = Mat::<f64>::zeros(p, p);
    let denom = (n - 1) as f64;
    let mut idx = 0;
    for a in 0..p {
        for b in a..p {
            let v = tri_sums[idx] / denom;
            cov[(a, b)] = v;
            cov[(b, a)] = v;
            idx += 1;
        }
    }
    cov
}

#[inline]
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[inline]
fn inv_softplus(y: f64) -> f64 {
    let y = y.max(1e-4);
    if y > 20.0 {
        y
    } else {
        (y.exp() - 1.0).ln()
    }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// ประมาณค่า CB-SEM ด้วย Maximum Likelihood (RAM Formulation)
pub fn fit_cb_ml<T: SemFloat>(
    ir: &CausalGraphIR,
    data: MatrixView<'_, T>,
    col_names: &[String],
    max_iter: usize,
) -> Result<CbResult, String> {
    let n_obs_rows = data.nrows();
    if n_obs_rows < 10 {
        return Err("CB-SEM requires at least 10 observations".into());
    }

    let col_map: HashMap<&str, usize> = col_names
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i))
        .collect();

    let p = ir.observed.len();
    let k = ir.latents.len();
    let m = p + k; // ขนาด RAM matrices = observed + latents

    let mut obs_cols = Vec::with_capacity(p);
    let mut node_idx: HashMap<&str, usize> = HashMap::new();
    for (i, name) in ir.observed.iter().enumerate() {
        let &c = col_map
            .get(name.as_str())
            .ok_or_else(|| format!("Observed variable '{}' not found in dataset", name))?;
        obs_cols.push(c);
        node_idx.insert(name.as_str(), i);
    }
    for (j, name) in ir.latents.iter().enumerate() {
        node_idx.insert(name.as_str(), p + j);
    }

    let s_obs = sample_covariance(data, &obs_cols);
    let s_llt = s_obs
        .cholesky(Side::Lower)
        .map_err(|_| "Sample covariance matrix is not positive definite".to_string())?;
    let s_l = s_llt.compute_l();
    let mut logdet_s = 0.0_f64;
    for i in 0..p {
        logdet_s += 2.0 * s_l[(i, i)].ln();
    }

    // สร้าง Parameter Vector theta และ Matrix Slots (รองรับ Fixed Params & Equality Labels)
    let mut slots: Vec<MatrixSlot> = Vec::new();
    let mut init_theta: Vec<f64> = Vec::new();
    let mut label_to_idx: HashMap<String, usize> = HashMap::new();

    let mut alloc_param = |spec: &ParamSpec, init_val: f64| -> (Option<usize>, f64) {
        match spec {
            ParamSpec::Fixed(v) => (None, *v),
            ParamSpec::Free => {
                let idx = init_theta.len();
                init_theta.push(init_val);
                (Some(idx), 0.0)
            }
            ParamSpec::Labeled(lbl) => {
                if let Some(&idx) = label_to_idx.get(lbl) {
                    (Some(idx), 0.0)
                } else {
                    let idx = init_theta.len();
                    init_theta.push(init_val);
                    label_to_idx.insert(lbl.clone(), idx);
                    (Some(idx), 0.0)
                }
            }
        }
    };

    // 1. Measurement Model (=~): ลูกศร Latent -> Indicator ใน A[ind, lat]
    // Auto Scale Identification: ถ้า Latent ไม่มี Fixed Loading ให้ล็อกตัวชี้วัดแรก = 1.0 (เหมือน lavaan)
    for blk in &ir.measurements {
        let l_idx = node_idx[blk.latent.as_str()];
        let has_fixed_loading = blk
            .terms
            .iter()
            .any(|t| matches!(t.param, ParamSpec::Fixed(_)));
        let has_fixed_var = ir.covariances.iter().any(|c| {
            c.lhs == blk.latent && c.rhs == blk.latent && matches!(c.param, ParamSpec::Fixed(_))
        });

        for (t_pos, t) in blk.terms.iter().enumerate() {
            let ind_idx = node_idx[t.name.as_str()];
            let effective_spec = if t_pos == 0 && !has_fixed_loading && !has_fixed_var {
                ParamSpec::Fixed(1.0)
            } else {
                t.param.clone()
            };
            let (pidx, fval) = alloc_param(&effective_spec, 0.7);
            slots.push(MatrixSlot {
                row: ind_idx,
                col: l_idx,
                is_symmetric_s: false,
                is_variance: false,
                param_idx: pidx,
                fixed_val: fval,
                lhs: blk.latent.clone(),
                op: "=~".into(),
                rhs: t.name.clone(),
            });
        }
    }

    // 2. Structural Model (~): ลูกศร from -> to ใน A[to, from]
    let mut has_incoming_structural: HashSet<usize> = HashSet::new();
    for e in &ir.structural_edges {
        let u = node_idx[e.from.as_str()];
        let v = node_idx[e.to.as_str()];
        has_incoming_structural.insert(v);
        let (pidx, fval) = alloc_param(&e.param, 0.2);
        slots.push(MatrixSlot {
            row: v,
            col: u,
            is_symmetric_s: false,
            is_variance: false,
            param_idx: pidx,
            fixed_val: fval,
            lhs: e.to.clone(),
            op: "~".into(),
            rhs: e.from.clone(),
        });
    }

    // 3. Explicit Covariances (~~)
    let mut explicit_diag: HashSet<usize> = HashSet::new();
    let mut explicit_pairs: HashSet<(usize, usize)> = HashSet::new();
    for c in &ir.covariances {
        let u = node_idx[c.lhs.as_str()];
        let v = node_idx[c.rhs.as_str()];
        let is_var = u == v;
        if is_var {
            explicit_diag.insert(u);
        } else {
            explicit_pairs.insert((u.min(v), u.max(v)));
        }
        let init_v = if is_var { inv_softplus(0.5) } else { 0.1 };
        let (pidx, fval) = alloc_param(&c.param, init_v);
        slots.push(MatrixSlot {
            row: u,
            col: v,
            is_symmetric_s: true,
            is_variance: is_var,
            param_idx: pidx,
            fixed_val: fval,
            lhs: c.lhs.clone(),
            op: "~~".into(),
            rhs: c.rhs.clone(),
        });
    }

    // 4. Default Variances ของทุก Observed & Latent ที่ยังไม่ได้ระบุ
    for i in 0..p {
        if !explicit_diag.contains(&i) {
            let init_v = inv_softplus((s_obs[(i, i)] * 0.5).max(0.05));
            let (pidx, fval) = alloc_param(&ParamSpec::Free, init_v);
            slots.push(MatrixSlot {
                row: i,
                col: i,
                is_symmetric_s: true,
                is_variance: true,
                param_idx: pidx,
                fixed_val: fval,
                lhs: ir.observed[i].clone(),
                op: "~~".into(),
                rhs: ir.observed[i].clone(),
            });
        }
    }
    for j in 0..k {
        let idx = p + j;
        if !explicit_diag.contains(&idx) {
            let (pidx, fval) = alloc_param(&ParamSpec::Free, inv_softplus(0.5));
            slots.push(MatrixSlot {
                row: idx,
                col: idx,
                is_symmetric_s: true,
                is_variance: true,
                param_idx: pidx,
                fixed_val: fval,
                lhs: ir.latents[j].clone(),
                op: "~~".into(),
                rhs: ir.latents[j].clone(),
            });
        }
    }

    // 5. Default Exogenous Latent Covariances (เหมือน lavaan: Exogenous latents สหสัมพันธ์กันได้)
    let exog_latents: Vec<usize> = (0..k)
        .map(|j| p + j)
        .filter(|idx| !has_incoming_structural.contains(idx))
        .collect();
    for a in 0..exog_latents.len() {
        for b in (a + 1)..exog_latents.len() {
            let u = exog_latents[a];
            let v = exog_latents[b];
            if !explicit_pairs.contains(&(u.min(v), u.max(v))) {
                let (pidx, fval) = alloc_param(&ParamSpec::Free, 0.05);
                slots.push(MatrixSlot {
                    row: u,
                    col: v,
                    is_symmetric_s: true,
                    is_variance: false,
                    param_idx: pidx,
                    fixed_val: fval,
                    lhs: ir.latents[u - p].clone(),
                    op: "~~".into(),
                    rhs: ir.latents[v - p].clone(),
                });
            }
        }
    }

    // Filter Matrix F = [I_p | 0] (ขนาด p x m)
    let f_mat = Mat::from_fn(p, m, |i, j| if i == j { 1.0 } else { 0.0 });
    let eye_m = Mat::<f64>::identity(m, m);

    let eval_ml = |theta: &[f64]| -> Result<(f64, Vec<f64>, Mat<f64>), String> {
        let mut a_mat = Mat::<f64>::zeros(m, m);
        let mut s_mat = Mat::<f64>::zeros(m, m);

        for slot in &slots {
            let val = match slot.param_idx {
                Some(idx) => {
                    if slot.is_variance {
                        softplus(theta[idx])
                    } else {
                        theta[idx]
                    }
                }
                None => slot.fixed_val,
            };
            if slot.is_symmetric_s {
                s_mat[(slot.row, slot.col)] = val;
                s_mat[(slot.col, slot.row)] = val;
            } else {
                a_mat[(slot.row, slot.col)] = val;
            }
        }

        let mut tape = Tape::new();
        let eye_id = tape.leaf(eye_m.clone());
        let a_id = tape.leaf(a_mat);
        let s_ram_id = tape.leaf(s_mat);
        let f_id = tape.leaf(f_mat.clone());
        let s_obs_id = tape.leaf(s_obs.clone());

        let i_minus_a = tape.sub(eye_id, a_id);
        let inv_ia = tape.mat_inv(i_minus_a)?;
        let f_inv = tape.matmul(f_id, inv_ia);
        let f_inv_t = tape.transpose(f_inv);
        let tmp = tape.matmul(f_inv, s_ram_id);
        let sigma_id = tape.matmul(tmp, f_inv_t);

        let logdet_sig = tape.logdet_pd(sigma_id)?;
        let sig_inv = tape.mat_inv(sigma_id)?;
        let s_sig_inv = tape.matmul(s_obs_id, sig_inv);
        let tr = tape.trace(s_sig_inv);
        let raw_loss = tape.add(logdet_sig, tr);

        let f_ml = (tape.value(raw_loss)[(0, 0)] - logdet_s - (p as f64)).max(0.0);
        let grads = tape.backward(raw_loss);
        let grad_a = &grads[a_id.0];
        let grad_s = &grads[s_ram_id.0];

        let mut g_theta = vec![0.0_f64; theta.len()];
        for slot in &slots {
            if let Some(idx) = slot.param_idx {
                if slot.is_symmetric_s {
                    if slot.is_variance {
                        g_theta[idx] += grad_s[(slot.row, slot.col)] * sigmoid(theta[idx]);
                    } else {
                        g_theta[idx] += grad_s[(slot.row, slot.col)] + grad_s[(slot.col, slot.row)];
                    }
                } else {
                    g_theta[idx] += grad_a[(slot.row, slot.col)];
                }
            }
        }

        Ok((f_ml, g_theta, tape.value(sigma_id).clone()))
    };

    let opt = minimize_lbfgs(&init_theta, max_iter, 1e-8, 10, |th| {
        let (f, g, _) = eval_ml(th)?;
        Ok((f, g))
    })?;

    let (f_ml, _, sigma_hat) = eval_ml(&opt.params)?;

    // คำนวณ Asymptotic Standard Errors จาก Hessian ของ F_ML ด้วย Central Differences บน Analytical Gradient
    let d = opt.params.len();
    let mut hess = Mat::<f64>::zeros(d, d);
    let eps = 1e-5;
    for j in 0..d {
        let mut tp = opt.params.clone();
        let mut tm = opt.params.clone();
        tp[j] += eps;
        tm[j] -= eps;
        if let (Ok((_, gp, _)), Ok((_, gm, _))) = (eval_ml(&tp), eval_ml(&tm)) {
            for i in 0..d {
                hess[(i, j)] = (gp[i] - gm[i]) / (2.0 * eps);
            }
        }
    }
    // Symmetrize และแปลงเป็น Information Matrix = ((N-1)/2) * H
    let scale_info = ((n_obs_rows - 1) as f64) * 0.5;
    let mut info = Mat::<f64>::zeros(d, d);
    for i in 0..d {
        for j in 0..d {
            info[(i, j)] = 0.5 * (hess[(i, j)] + hess[(j, i)]) * scale_info;
        }
        info[(i, i)] += 1e-9; // Ridge stabilization
    }
    let cov_theta = info.partial_piv_lu().solve(&Mat::<f64>::identity(d, d));

    let params_out: Vec<CbParamEstimate> = slots
        .iter()
        .map(|slot| match slot.param_idx {
            Some(idx) => {
                let raw_se = cov_theta[(idx, idx)].max(0.0).sqrt();
                let (est, se) = if slot.is_variance {
                    // Delta method: SE(softplus(theta)) = sigmoid(theta) * SE(theta)
                    (softplus(opt.params[idx]), raw_se * sigmoid(opt.params[idx]))
                } else {
                    (opt.params[idx], raw_se)
                };
                CbParamEstimate {
                    lhs: slot.lhs.clone(),
                    op: slot.op.clone(),
                    rhs: slot.rhs.clone(),
                    estimate: est,
                    se,
                    fixed: false,
                }
            }
            None => CbParamEstimate {
                lhs: slot.lhs.clone(),
                op: slot.op.clone(),
                rhs: slot.rhs.clone(),
                estimate: slot.fixed_val,
                se: 0.0,
                fixed: true,
            },
        })
        .collect();

    // คำนวณ Fit Indices มาตรฐานสากล (Chi2, df, CFI, TLI, RMSEA, SRMR)
    let n_moments = p * (p + 1) / 2;
    let df = n_moments.saturating_sub(d);
    let chi_sq = ((n_obs_rows - 1) as f64) * f_ml;

    // Baseline Independence Model (Diagonal of S)
    let mut logdet_diag = 0.0_f64;
    for i in 0..p {
        logdet_diag += s_obs[(i, i)].max(1e-15).ln();
    }
    let f_null = (logdet_diag - logdet_s).max(0.0);
    let chi_null = ((n_obs_rows - 1) as f64) * f_null;
    let df_null = p * (p - 1) / 2;

    let cfi = if (chi_null - df_null as f64) > 1e-12 {
        (1.0 - (chi_sq - df as f64).max(0.0) / (chi_null - df_null as f64)).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let tli = if df > 0 && df_null > 0 && (chi_null / df_null as f64 - 1.0).abs() > 1e-12 {
        (chi_null / df_null as f64 - chi_sq / df as f64) / (chi_null / df_null as f64 - 1.0)
    } else {
        1.0
    };
    let rmsea = if df > 0 {
        (((chi_sq - df as f64).max(0.0)) / ((df as f64) * ((n_obs_rows - 1) as f64))).sqrt()
    } else {
        0.0
    };

    // SRMR (Standardized Root Mean Square Residual)
    let mut srmr_sum = 0.0_f64;
    let mut srmr_cnt = 0usize;
    for i in 0..p {
        for j in 0..=i {
            let denom = (s_obs[(i, i)] * s_obs[(j, j)]).sqrt().max(1e-12);
            let diff = (s_obs[(i, j)] - sigma_hat[(i, j)]) / denom;
            srmr_sum += diff * diff;
            srmr_cnt += 1;
        }
    }
    let srmr = (srmr_sum / (srmr_cnt as f64)).sqrt();

    Ok(CbResult {
        converged: opt.converged,
        iterations: opt.iterations,
        f_ml,
        params: params_out,
        fit: FitIndices {
            chi_square: chi_sq,
            df,
            cfi,
            tli,
            rmsea,
            srmr,
        },
    })
}
