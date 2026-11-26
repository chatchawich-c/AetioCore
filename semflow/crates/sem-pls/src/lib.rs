use faer::prelude::*;
use rayon::prelude::*;
use sem_dsl::{CausalGraphIR, MeasurementKind};
use sem_traits::{MatrixView, SemFloat};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct PathEstimate {
    pub from: String,
    pub to: String,
    pub estimate: f64,
    pub se: Option<f64>,
    pub t_stat: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct ConstructMetrics {
    pub latent: String,
    pub mode: String,
    pub rho_a: f64,
    pub ave: f64,
    pub cr: f64,
    pub r_squared: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct PlsResult {
    pub consistent: bool,
    pub iterations: usize,
    pub converged: bool,
    pub paths: Vec<PathEstimate>,
    pub loadings: Vec<(String, String, f64)>,
    pub weights: Vec<(String, String, f64)>,
    pub metrics: Vec<ConstructMetrics>,
    pub htmt: Vec<(String, String, f64)>,
}

struct CorePlsOut {
    iterations: usize,
    converged: bool,
    path_vals: Vec<(usize, usize, f64)>,
    r2_map: HashMap<usize, f64>,
    loadings: Vec<Vec<f64>>,
    weights: Vec<Vec<f64>>,
    rho_a: Vec<f64>,
}

/// คำนวณ Sample Correlation Matrix R (p x p) แบบ Zero-Copy (ขนานด้วย rayon เมื่อไม่ใช้ Bootstrap)
fn sample_correlation<T: SemFloat>(
    data: MatrixView<'_, T>,
    selected_cols: &[usize],
    row_indices: Option<&[usize]>,
) -> Mat<f64> {
    let p = selected_cols.len();
    let tri_len = p * (p + 1) / 2;

    let tri_sums = match row_indices {
        None => {
            let n = data.nrows();
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
                            local[j] += data.get(i, selected_cols[j]);
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

            (0..n_chunks)
                .into_par_iter()
                .map(|c| {
                    let start = c * chunk_size;
                    let end = (start + chunk_size).min(n);
                    let mut local = vec![0.0_f64; tri_len];
                    let mut row_d = vec![0.0_f64; p];
                    for i in start..end {
                        for a in 0..p {
                            row_d[a] = data.get(i, selected_cols[a]) - means[a];
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
                )
        }
        Some(indices) => {
            let n = indices.len();
            let mut means = vec![0.0_f64; p];
            for &r in indices {
                for j in 0..p {
                    means[j] += data.get(r, selected_cols[j]);
                }
            }
            for m in &mut means {
                *m /= n as f64;
            }

            let mut local = vec![0.0_f64; tri_len];
            let mut row_d = vec![0.0_f64; p];
            for &r in indices {
                for a in 0..p {
                    row_d[a] = data.get(r, selected_cols[a]) - means[a];
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
        }
    };

    // ดึง sum-of-squares แนวทแยงเพื่อปรับเป็น Correlation Matrix
    let mut std_sq = vec![0.0_f64; p];
    let mut idx = 0;
    for a in 0..p {
        std_sq[a] = tri_sums[idx].sqrt().max(1e-12);
        idx += p - a;
    }

    let mut corr = Mat::<f64>::zeros(p, p);
    let mut idx = 0;
    for a in 0..p {
        for b in a..p {
            if a == b {
                corr[(a, a)] = 1.0;
            } else {
                let v = (tri_sums[idx] / (std_sq[a] * std_sq[b])).clamp(-1.0, 1.0);
                corr[(a, b)] = v;
                corr[(b, a)] = v;
            }
            idx += 1;
        }
    }
    corr
}

/// คำนวณ Dijkstra's rho_A สำหรับ PLSc (Consistent PLS)
fn compute_rho_a(w: &[f64], s_block: &Mat<f64>) -> f64 {
    let m = w.len();
    if m <= 1 {
        return 1.0;
    }
    let ww: f64 = w.iter().map(|&x| x * x).sum();
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for i in 0..m {
        for j in 0..m {
            if i != j {
                num += w[i] * s_block[(i, j)] * w[j];
                den += (w[i] * w[j]).powi(2);
            }
        }
    }
    if den.abs() < 1e-15 || num <= 0.0 {
        return 1.0;
    }
    let rho = (ww * ww) * (num / den);
    rho.clamp(0.05, 1.0)
}

#[inline]
fn latent_corr(r_obs: &Mat<f64>, b1: &[usize], w1: &[f64], b2: &[usize], w2: &[f64]) -> f64 {
    let mut s = 0.0_f64;
    for (a, &c1) in b1.iter().enumerate() {
        let wa = w1[a];
        for (b, &c2) in b2.iter().enumerate() {
            s += wa * r_obs[(c1, c2)] * w2[b];
        }
    }
    s
}

/// รันอัลกอริทึม Wold PLS บน Sample Correlation Matrix R (p x p) โดยไม่แตะข้อมูล N แถวในลูป
fn run_pls_core_from_corr(
    r_obs: &Mat<f64>,
    blocks: &[Vec<usize>],
    kinds: &[MeasurementKind],
    edges: &[(usize, usize)],
    consistent: bool,
    max_iter: usize,
    tol: f64,
) -> CorePlsOut {
    let k_count = blocks.len();

    let block_corrs: Vec<Mat<f64>> = blocks
        .iter()
        .map(|b| {
            let m = b.len();
            Mat::from_fn(m, m, |a, c| r_obs[(b[a], b[c])])
        })
        .collect();

    let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); k_count];
    for &(u, v) in edges {
        if !neighbors[u].contains(&v) {
            neighbors[u].push(v);
        }
        if !neighbors[v].contains(&u) {
            neighbors[v].push(u);
        }
    }

    // เริ่มต้น weights และ normalize ให้ w_k^T R_{kk} w_k = 1
    let mut weights: Vec<Vec<f64>> = (0..k_count)
        .map(|k| {
            let m = blocks[k].len();
            let w0 = vec![1.0 / (m as f64).sqrt(); m];
            let quad = latent_corr(r_obs, &blocks[k], &w0, &blocks[k], &w0)
                .sqrt()
                .max(1e-12);
            w0.into_iter().map(|v| v / quad).collect()
        })
        .collect();

    let mut converged = false;
    let mut iters = 0;

    for iter in 1..=max_iter {
        iters = iter;
        let mut next_weights = Vec::with_capacity(k_count);
        let mut max_diff = 0.0_f64;

        for k in 0..k_count {
            let m = blocks[k].len();
            let nbrs = &neighbors[k];

            // 1. Inner Centroid Signs e_{kj} = sign(Corr(Y_k, Y_j))
            let signs: Vec<f64> = nbrs
                .iter()
                .map(|&j| {
                    let r = latent_corr(r_obs, &blocks[k], &weights[k], &blocks[j], &weights[j]);
                    if r >= 0.0 {
                        1.0
                    } else {
                        -1.0
                    }
                })
                .collect();

            // 2. Var(Z_k_raw) = sum_{a,b} e_{ka} e_{kb} Corr(Y_a, Y_b)
            let mut var_z = 0.0_f64;
            for (ia, &ja) in nbrs.iter().enumerate() {
                for (ib, &jb) in nbrs.iter().enumerate() {
                    var_z += signs[ia]
                        * signs[ib]
                        * latent_corr(r_obs, &blocks[ja], &weights[ja], &blocks[jb], &weights[jb]);
                }
            }
            let std_z = var_z.sqrt().max(1e-12);

            // 3. r_{X_k, Z_k} = Cov(X_k, Z_k_raw) / std_z
            let mut r_xz = Mat::<f64>::zeros(m, 1);
            for (a, &col_a) in blocks[k].iter().enumerate() {
                let mut cov = 0.0_f64;
                for (ij, &j) in nbrs.iter().enumerate() {
                    let mut dot_j = 0.0_f64;
                    for (b, &col_b) in blocks[j].iter().enumerate() {
                        dot_j += r_obs[(col_a, col_b)] * weights[j][b];
                    }
                    cov += signs[ij] * dot_j;
                }
                r_xz[(a, 0)] = cov / std_z;
            }

            // 4. Outer Weight Update (Mode A vs Mode B)
            let new_w_vec: Vec<f64> = match kinds[k] {
                MeasurementKind::Reflective => (0..m).map(|a| r_xz[(a, 0)]).collect(),
                MeasurementKind::Formative => {
                    let sol = block_corrs[k].qr().solve(&r_xz);
                    (0..m).map(|a| sol[(a, 0)]).collect()
                }
            };

            // Normalize w_k ให้ w_k^T R_{kk} w_k = 1
            let scale = latent_corr(r_obs, &blocks[k], &new_w_vec, &blocks[k], &new_w_vec)
                .sqrt()
                .max(1e-12);

            let mut norm_w = Vec::with_capacity(m);
            for a in 0..m {
                let nw = new_w_vec[a] / scale;
                max_diff = max_diff.max((nw - weights[k][a]).abs());
                norm_w.push(nw);
            }
            next_weights.push(norm_w);
        }

        weights = next_weights;
        if max_diff < tol {
            converged = true;
            break;
        }
    }

    // Sign Alignment (ให้ผลรวม weight เป็นบวก)
    for w in &mut weights {
        if w.iter().sum::<f64>() < 0.0 {
            for v in w.iter_mut() {
                *v = -*v;
            }
        }
    }

    // Dijkstra's rho_A
    let rho_a: Vec<f64> = (0..k_count)
        .map(|k| {
            if consistent && kinds[k] == MeasurementKind::Reflective {
                compute_rho_a(&weights[k], &block_corrs[k])
            } else {
                1.0
            }
        })
        .collect();

    // Latent Correlation Matrix
    let mut r_lat = Mat::<f64>::zeros(k_count, k_count);
    for i in 0..k_count {
        for j in 0..k_count {
            if i == j {
                r_lat[(i, j)] = 1.0;
            } else {
                let raw = latent_corr(r_obs, &blocks[i], &weights[i], &blocks[j], &weights[j]);
                let adj = if consistent {
                    (raw / (rho_a[i] * rho_a[j]).sqrt()).clamp(-0.999, 0.999)
                } else {
                    raw
                };
                r_lat[(i, j)] = adj;
            }
        }
    }

    // Outer Loadings: R_{kk} * w_k (หรือ Consistent Loading เมื่อเปิด PLSc)
    let mut loadings = Vec::with_capacity(k_count);
    for k in 0..k_count {
        let m = blocks[k].len();
        if consistent && kinds[k] == MeasurementKind::Reflective && m > 1 {
            let ww: f64 = weights[k].iter().map(|&x| x * x).sum();
            let factor = rho_a[k].sqrt() / ww.max(1e-12);
            loadings.push(weights[k].iter().map(|&w| w * factor).collect());
        } else {
            let lvec: Vec<f64> = (0..m)
                .map(|a| {
                    (0..m)
                        .map(|b| block_corrs[k][(a, b)] * weights[k][b])
                        .sum()
                })
                .collect();
            loadings.push(lvec);
        }
    }

    // Path Coefficients และ R^2 จาก r_lat ด้วย faer QR
    let mut path_vals = Vec::new();
    let mut r2_map = HashMap::new();
    for target in 0..k_count {
        let preds: Vec<usize> = edges
            .iter()
            .filter(|&&(_, to)| to == target)
            .map(|&(from, _)| from)
            .collect();
        if preds.is_empty() {
            continue;
        }
        let m = preds.len();
        let mut r_xx = Mat::<f64>::zeros(m, m);
        let mut r_xy = Mat::<f64>::zeros(m, 1);
        for a in 0..m {
            r_xy[(a, 0)] = r_lat[(preds[a], target)];
            for b in 0..m {
                r_xx[(a, b)] = r_lat[(preds[a], preds[b])];
            }
        }
        let beta = r_xx.qr().solve(&r_xy);
        let mut r2 = 0.0_f64;
        for a in 0..m {
            let b_val = beta[(a, 0)];
            r2 += b_val * r_xy[(a, 0)];
            path_vals.push((preds[a], target, b_val));
        }
        r2_map.insert(target, r2);
    }

    CorePlsOut {
        iterations: iters,
        converged,
        path_vals,
        r2_map,
        loadings,
        weights,
        rho_a,
    }
}

/// SplitMix64 PRNG แบบเบาสำหรับ Parallel Bootstrap (ไม่เพิ่ม dependency)
#[inline]
fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// รัน PLS-SEM / PLSc พร้อมดัชนี AVE, CR, HTMT และ Parallel Bootstrap (rayon)
pub fn fit_pls<T: SemFloat>(
    ir: &CausalGraphIR,
    data: MatrixView<'_, T>,
    col_names: &[String],
    consistent: bool,
    n_boot: usize,
    max_iter: usize,
    tol: f64,
) -> Result<PlsResult, String> {
    if data.nrows() < 5 {
        return Err("PLS-SEM requires at least 5 observations".into());
    }
    let col_map: HashMap<&str, usize> = col_names
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i))
        .collect();

    let mut selected_cols = Vec::new();
    let mut local_col_map: HashMap<&str, usize> = HashMap::new();
    for obs in &ir.observed {
        let &global_idx = col_map
            .get(obs.as_str())
            .ok_or_else(|| format!("Column '{}' not found in dataset", obs))?;
        local_col_map.insert(obs.as_str(), selected_cols.len());
        selected_cols.push(global_idx);
    }

    let k_count = ir.latents.len();
    let latent_idx: HashMap<&str, usize> = ir
        .latents
        .iter()
        .enumerate()
        .map(|(i, l)| (l.as_str(), i))
        .collect();

    let mut blocks = Vec::with_capacity(k_count);
    let mut kinds = Vec::with_capacity(k_count);
    for l in &ir.latents {
        let blk = ir
            .measurements
            .iter()
            .find(|b| b.latent == *l)
            .ok_or_else(|| format!("Latent '{}' missing measurement block", l))?;
        let mut cols = Vec::new();
        for t in &blk.terms {
            let &lc = local_col_map.get(t.name.as_str()).ok_or_else(|| {
                format!(
                    "Indicator '{}' in latent '{}' must be observed for Stage-1 PLS",
                    t.name, l
                )
            })?;
            cols.push(lc);
        }
        blocks.push(cols);
        kinds.push(blk.kind);
    }

    let edges: Vec<(usize, usize)> = ir
        .structural_edges
        .iter()
        .filter_map(|e| Some((*latent_idx.get(e.from.as_str())?, *latent_idx.get(e.to.as_str())?)))
        .collect();

    let r_obs = sample_correlation(data, &selected_cols, None);
    let base = run_pls_core_from_corr(&r_obs, &blocks, &kinds, &edges, consistent, max_iter, tol);

    let boot_ses: Option<Vec<f64>> = if n_boot > 1 {
        let n = data.nrows();
        let n_paths = base.path_vals.len();
        let samples: Vec<Vec<f64>> = (0..n_boot)
            .into_par_iter()
            .map(|b_idx| {
                let mut seed = 0x5EED_CAFE_u64 ^ ((b_idx as u64).wrapping_mul(0x9E37_79B9));
                let row_idx: Vec<usize> = (0..n)
                    .map(|_| (next_u64(&mut seed) as usize) % n)
                    .collect();
                let rb = sample_correlation(data, &selected_cols, Some(&row_idx));
                let out = run_pls_core_from_corr(
                    &rb, &blocks, &kinds, &edges, consistent, max_iter, tol,
                );
                out.path_vals.into_iter().map(|(_, _, v)| v).collect()
            })
            .collect();

        let mut ses = vec![0.0_f64; n_paths];
        for p in 0..n_paths {
            let mean: f64 = samples.iter().map(|s| s[p]).sum::<f64>() / (n_boot as f64);
            let var: f64 = samples
                .iter()
                .map(|s| (s[p] - mean).powi(2))
                .sum::<f64>()
                / ((n_boot - 1) as f64);
            ses[p] = var.sqrt();
        }
        Some(ses)
    } else {
        None
    };

    let paths: Vec<PathEstimate> = base
        .path_vals
        .iter()
        .enumerate()
        .map(|(i, &(u, v, est))| {
            let se = boot_ses.as_ref().map(|s| s[i]);
            let t_stat = se.map(|s| if s > 1e-12 { est / s } else { f64::NAN });
            PathEstimate {
                from: ir.latents[u].clone(),
                to: ir.latents[v].clone(),
                estimate: est,
                se,
                t_stat,
            }
        })
        .collect();

    let mut loadings_out = Vec::new();
    let mut weights_out = Vec::new();
    let mut metrics = Vec::with_capacity(k_count);

    for k in 0..k_count {
        let mut sum_l = 0.0_f64;
        let mut sum_l2 = 0.0_f64;
        let mut sum_err = 0.0_f64;
        for (a, &lc) in blocks[k].iter().enumerate() {
            let l_val = base.loadings[k][a];
            let w_val = base.weights[k][a];
            let ind_name = col_names[selected_cols[lc]].clone();
            loadings_out.push((ir.latents[k].clone(), ind_name.clone(), l_val));
            weights_out.push((ir.latents[k].clone(), ind_name, w_val));
            sum_l += l_val;
            sum_l2 += l_val * l_val;
            sum_err += (1.0 - l_val * l_val).max(0.0);
        }
        let m = blocks[k].len() as f64;
        let ave = sum_l2 / m;
        let cr = (sum_l * sum_l) / ((sum_l * sum_l) + sum_err).max(1e-12);
        metrics.push(ConstructMetrics {
            latent: ir.latents[k].clone(),
            mode: match kinds[k] {
                MeasurementKind::Reflective => "Mode A (Reflective)".into(),
                MeasurementKind::Formative => "Mode B (Formative)".into(),
            },
            rho_a: base.rho_a[k],
            ave,
            cr,
            r_squared: base.r2_map.get(&k).copied(),
        });
    }

    // HTMT คำนวณตรงจาก r_obs
    let mut htmt = Vec::new();
    for i in 0..k_count {
        for j in (i + 1)..k_count {
            if kinds[i] != MeasurementKind::Reflective
                || kinds[j] != MeasurementKind::Reflective
                || blocks[i].len() < 2
                || blocks[j].len() < 2
            {
                continue;
            }
            let mut hetero = 0.0_f64;
            for &c1 in &blocks[i] {
                for &c2 in &blocks[j] {
                    hetero += r_obs[(c1, c2)].abs();
                }
            }
            hetero /= (blocks[i].len() * blocks[j].len()) as f64;

            let mono_avg = |blk: &[usize]| -> f64 {
                let mut s = 0.0_f64;
                let mut cnt = 0usize;
                for a in 0..blk.len() {
                    for b in (a + 1)..blk.len() {
                        s += r_obs[(blk[a], blk[b])].abs();
                        cnt += 1;
                    }
                }
                s / (cnt as f64)
            };
            let denom = (mono_avg(&blocks[i]) * mono_avg(&blocks[j])).sqrt().max(1e-12);
            htmt.push((ir.latents[i].clone(), ir.latents[j].clone(), hetero / denom));
        }
    }

    Ok(PlsResult {
        consistent,
        iterations: base.iterations,
        converged: base.converged,
        paths,
        loadings: loadings_out,
        weights: weights_out,
        metrics,
        htmt,
    })
}
