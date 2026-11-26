pub mod spatial_purify;
pub mod spline;

use rayon::prelude::*;
use sem_autodiff::minimize_lbfgs;
use sem_dsl::{CausalGraphIR, Constraint};
use sem_traits::{Calibrator, MatrixView, SemFloat, StructuralOperator};
use spatial_purify::{CsrMatrix, Purified2DInteraction, SpatialCoupling};
use spline::{EdgeOperator, SplineBasis};
use std::collections::HashMap;

/// Split-Conformal Calibrator (implement sem_traits::Calibrator)
#[derive(Debug, Clone, Default)]
pub struct SplitConformalCalibrator {
    pub alpha: f64,
    pub quantile_q: f64,
}

impl Calibrator for SplitConformalCalibrator {
    fn calibrate(&mut self, residuals: &[f64], alpha: f64) {
        self.alpha = alpha.clamp(1e-4, 0.5);
        if residuals.is_empty() {
            self.quantile_q = 0.0;
            return;
        }
        let mut abs_res: Vec<f64> = residuals.iter().map(|r| r.abs()).collect();
        abs_res.sort_by(|a, b| a.total_cmp(b));
        let n = abs_res.len() as f64;
        // Finite-sample distribution-free conformal quantile: ceil((n+1)*(1-alpha))/n
        let rank = ((n + 1.0) * (1.0 - self.alpha)).ceil() as usize;
        let idx = rank.saturating_sub(1).min(abs_res.len() - 1);
        self.quantile_q = abs_res[idx];
    }

    fn half_width(&self) -> f64 {
        self.quantile_q
    }
}

#[derive(Debug, Clone)]
pub struct FittedEdge {
    pub from: String,
    pub to: String,
    pub operator: EdgeOperator,
    pub params: Vec<f64>,
    /// ค่าเฉลี่ยของ phi(z) บนชุดเทรน ใช้สำหรับ Centering ให้กราฟ 1D มีค่าเฉลี่ย = 0 เหมือน EBM
    pub mean_shift: f64,
    pub mean_elasticity: f64,
}

#[derive(Debug, Clone)]
pub struct MeasurementAnchor {
    pub latent: String,
    pub indicator_names: Vec<String>,
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
    pub weights: Vec<f64>, // หน่วยความแปรปรวน = 1, ทิศทางบวก (Anchored)
}

#[derive(Debug, Clone)]
pub struct WhiteBoxModel {
    pub ir: CausalGraphIR,
    pub anchors: HashMap<String, MeasurementAnchor>,
    pub edges: Vec<FittedEdge>,
    pub intercepts: HashMap<String, f64>,
    pub r_squared: HashMap<String, f64>,
    pub calibrators: HashMap<String, SplitConformalCalibrator>,
    pub interactions: Vec<Purified2DInteraction>,
    pub spatial: Option<SpatialCoupling>,
}

#[derive(Debug, Clone)]
pub struct EdgeCurve {
    pub from: String,
    pub to: String,
    pub edge_type: String,
    pub x_grid: Vec<f64>,
    pub y_centered: Vec<f64>,
    pub local_elasticity: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct InterventionSummary {
    pub intervened_var: String,
    pub delta: f64,
    pub target_var: String,
    pub total_effect: f64,
    pub direct_effect: f64,
    pub indirect_effect: f64,
    pub conformal_half_width: f64,
}

/// สร้าง Anchored Latent Scores จากตัวชี้วัด (First Principal Component / Equal-Weight Correlation Anchor)
fn fit_measurement_anchors<T: SemFloat>(
    ir: &CausalGraphIR,
    data: MatrixView<'_, T>,
    col_names: &[String],
) -> Result<(HashMap<String, MeasurementAnchor>, HashMap<String, Vec<f64>>), String> {
    let n = data.nrows();
    let col_map: HashMap<&str, usize> = col_names
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i))
        .collect();

    let mut anchors = HashMap::new();
    let mut scores = HashMap::new();

    for blk in &ir.measurements {
        let m = blk.terms.len();
        let mut cols = Vec::with_capacity(m);
        let mut ind_names = Vec::with_capacity(m);
        for t in &blk.terms {
            let &c = col_map
                .get(t.name.as_str())
                .ok_or_else(|| format!("Indicator '{}' not found", t.name))?;
            cols.push(c);
            ind_names.push(t.name.clone());
        }

        // Standardize แต่ละ indicator (ทนทานต่อค่าว่างโดยใช้ mean fallback เฉพาะเซลล์ว่าง)
        let mut means = vec![0.0_f64; m];
        let mut stds = vec![1.0_f64; m];
        let mut z_cols = vec![vec![0.0_f64; n]; m];

        for (j, &c) in cols.iter().enumerate() {
            let mut sum = 0.0_f64;
            let mut cnt = 0usize;
            for r in 0..n {
                let v = data.get(r, c);
                if v.is_finite() {
                    sum += v;
                    cnt += 1;
                }
            }
            let mu = if cnt > 0 { sum / (cnt as f64) } else { 0.0 };
            let mut sq = 0.0_f64;
            for r in 0..n {
                let v = data.get(r, c);
                if v.is_finite() {
                    sq += (v - mu).powi(2);
                }
            }
            let sd = if cnt > 1 { (sq / (cnt as f64)).sqrt().max(1e-12) } else { 1.0 };
            means[j] = mu;
            stds[j] = sd;
            for r in 0..n {
                let v = data.get(r, c);
                z_cols[j][r] = if v.is_finite() { (v - mu) / sd } else { 0.0 };
            }
        }

        // Power Iteration หา First Principal Component ของ block เพื่อกรอง Measurement Noise
        let mut w = vec![1.0 / (m as f64).sqrt(); m];
        let mut latent_vec = vec![0.0_f64; n];
        for _ in 0..15 {
            latent_vec.fill(0.0);
            for j in 0..m {
                for r in 0..n {
                    latent_vec[r] += z_cols[j][r] * w[j];
                }
            }
            // คำนวณ w ใหม่ ~ Z^T * latent
            let mut w_new = vec![0.0_f64; m];
            for j in 0..m {
                let mut s = 0.0_f64;
                for r in 0..n {
                    s += z_cols[j][r] * latent_vec[r];
                }
                w_new[j] = s / (n as f64);
            }
            // Anchor sign ให้ตัวชี้วัดแรกมีน้ำหนักเป็นบวกเสมอ
            if w_new[0] < 0.0 {
                for x in &mut w_new {
                    *x = -*x;
                }
            }
            let norm = w_new.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
            for x in &mut w_new {
                *x /= norm;
            }
            w = w_new;
        }

        // สร้างคะแนนตัวแปรแฝงสุดท้ายพร้อม Scale ให้ Var = 1.0
        latent_vec.fill(0.0);
        for j in 0..m {
            for r in 0..n {
                latent_vec[r] += z_cols[j][r] * w[j];
            }
        }
        let l_var = latent_vec.iter().map(|v| v * v).sum::<f64>() / (n as f64);
        let l_sd = l_var.sqrt().max(1e-12);
        for x in &mut w {
            *x /= l_sd;
        }
        for v in &mut latent_vec {
            *v /= l_sd;
        }

        anchors.insert(
            blk.latent.clone(),
            MeasurementAnchor {
                latent: blk.latent.clone(),
                indicator_names: ind_names,
                means,
                stds,
                weights: w,
            },
        );
        scores.insert(blk.latent.clone(), latent_vec);
    }

    Ok((anchors, scores))
}

/// แปลงข้อมูลใหม่ (Test / Calibration) เป็น Latent Scores โดยใช้ Anchors เดิมจากชุดเทรน
pub fn transform_latents<T: SemFloat>(
    anchors: &HashMap<String, MeasurementAnchor>,
    data: MatrixView<'_, T>,
    col_names: &[String],
) -> Result<HashMap<String, Vec<f64>>, String> {
    let n = data.nrows();
    let col_map: HashMap<&str, usize> = col_names
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i))
        .collect();

    let mut out = HashMap::new();
    for (lat, anc) in anchors {
        let mut s = vec![0.0_f64; n];
        for (j, ind) in anc.indicator_names.iter().enumerate() {
            let &c = col_map
                .get(ind.as_str())
                .ok_or_else(|| format!("Column '{}' missing in input data", ind))?;
            let mu = anc.means[j];
            let sd = anc.stds[j];
            let w = anc.weights[j];
            for r in 0..n {
                let v = data.get(r, c);
                if v.is_finite() {
                    s[r] += ((v - mu) / sd) * w;
                }
            }
        }
        out.insert(lat.clone(), s);
    }
    Ok(out)
}

/// สร้าง EdgeOperator ตาม metadata ในสูตร เช่น [edge=ispline, sign=+1, knots=6]
fn build_edge_operator(constraints: &[Constraint], parent_vals: &[f64], default_knots: usize) -> EdgeOperator {
    let mut edge_kind = "ispline".to_string();
    let mut sign = 1.0_f64;
    let mut has_mono = false;
    let mut knots = default_knots;

    for c in constraints {
        match c {
            Constraint::EdgeType(et) => edge_kind = et.to_lowercase(),
            Constraint::Sign(s) => {
                sign = if *s < 0 { -1.0 } else { 1.0 };
                has_mono = true;
            }
            Constraint::Monotonic(m) => has_mono = *m,
            Constraint::Custom(k, v) if k == "knots" => {
                if let Ok(parsed) = v.parse::<usize>() {
                    knots = parsed.clamp(2, 20);
                }
            }
            _ => {}
        }
    }

    match edge_kind.as_str() {
        "linear" => EdgeOperator::Linear,
        "bspline" if !has_mono => EdgeOperator::FreeBSpline(SplineBasis::from_quantiles(parent_vals, knots, 3)),
        _ => EdgeOperator::MonotonicISpline {
            sign,
            basis: SplineBasis::from_quantiles(parent_vals, knots, 3),
        },
    }
}

/// เทรน Edge-Level White-Box SEM ทั้งกราฟ DAG ตามลำดับ Topological Order
pub fn fit_whitebox_sem<T: SemFloat>(
    formula: &str,
    data: MatrixView<'_, T>,
    col_names: &[String],
    default_knots: usize,
    smooth_lambda: f64,
    max_iter: usize,
) -> Result<WhiteBoxModel, String> {
    let ir = sem_dsl::parse_formula(formula)?;
    let (anchors, latent_scores) = fit_measurement_anchors(&ir, data, col_names)?;
    let n = data.nrows();

    let mut fitted_edges = Vec::new();
    let mut intercepts = HashMap::new();
    let mut r_squared = HashMap::new();

    // เทรนทีละ Endogenous Latent ตามลำดับ DAG
    for target in &ir.topo_order {
        let incoming: Vec<_> = ir
            .structural_edges
            .iter()
            .filter(|e| e.to == *target)
            .collect();
        if incoming.is_empty() {
            continue;
        }

        let y = latent_scores
            .get(target)
            .ok_or_else(|| format!("Target latent '{}' missing", target))?;

        let mut ops = Vec::with_capacity(incoming.len());
        let mut parent_vecs = Vec::with_capacity(incoming.len());
        let mut offsets = Vec::with_capacity(incoming.len());
        let mut init_theta = vec![0.0_f64]; // theta[0] = intercept b_j

        for e in &incoming {
            let pv = latent_scores
                .get(&e.from)
                .ok_or_else(|| format!("Parent latent '{}' missing", e.from))?;
            let op = build_edge_operator(&e.constraints, pv, default_knots);
            offsets.push(init_theta.len());
            init_theta.extend(op.init_params());
            ops.push(op);
            parent_vecs.push(pv);
        }

        // Precompute spline basis matrix หนึ่งครั้งสำหรับแต่ละเส้นเชื่อม เพื่อความเร็วระดับมิลลิวินาทีใน L-BFGS
        let precomputed_basis: Vec<Option<Vec<Vec<f64>>>> = ops
            .iter()
            .zip(parent_vecs.iter())
            .map(|(op, pv)| match op {
                EdgeOperator::Linear => None,
                EdgeOperator::FreeBSpline(basis) | EdgeOperator::MonotonicISpline { basis, .. } => {
                    let rows: Vec<Vec<f64>> = pv
                        .par_iter()
                        .map(|&x| basis.eval_ispline_and_mspline(x).0)
                        .collect();
                    Some(rows)
                }
            })
            .collect();

        let n_f = n as f64;
        let opt = minimize_lbfgs(&init_theta, max_iter, 1e-7, 10, |theta| {
            let b0 = theta[0];
            let mut preds = vec![b0; n];

            for (e_idx, op) in ops.iter().enumerate() {
                let off = offsets[e_idx];
                let pv = parent_vecs[e_idx];
                match op {
                    EdgeOperator::Linear => {
                        let beta = theta[off];
                        for r in 0..n {
                            preds[r] += beta * pv[r];
                        }
                    }
                    EdgeOperator::FreeBSpline(basis) => {
                        let w0 = theta[off];
                        let c = &theta[(off + 1)..(off + 1 + basis.n_basis)];
                        let bmat = precomputed_basis[e_idx].as_ref().unwrap();
                        for r in 0..n {
                            let mut v = w0 * pv[r];
                            for (k, &ck) in c.iter().enumerate() {
                                v += ck * bmat[r][k];
                            }
                            preds[r] += v;
                        }
                    }
                    EdgeOperator::MonotonicISpline { sign, basis } => {
                        let w0 = spline::softplus(theta[off]);
                        let w: Vec<f64> = theta[(off + 1)..(off + 1 + basis.n_basis)]
                            .iter()
                            .map(|&a| spline::softplus(a))
                            .collect();
                        let bmat = precomputed_basis[e_idx].as_ref().unwrap();
                        for r in 0..n {
                            let mut v = w0 * pv[r];
                            for (k, &wk) in w.iter().enumerate() {
                                v += wk * bmat[r][k];
                            }
                            preds[r] += sign * v;
                        }
                    }
                }
            }

            // MSE Loss + Analytical Gradients
            let mut mse = 0.0_f64;
            let mut res = vec![0.0_f64; n];
            let mut grad = vec![0.0_f64; theta.len()];
            for r in 0..n {
                let err = preds[r] - y[r];
                res[r] = err;
                mse += err * err;
                grad[0] += err;
            }
            mse /= n_f;
            grad[0] *= 2.0 / n_f;

            let scale = 2.0 / n_f;
            for (e_idx, op) in ops.iter().enumerate() {
                let off = offsets[e_idx];
                let pv = parent_vecs[e_idx];
                match op {
                    EdgeOperator::Linear => {
                        let g: f64 = (0..n).map(|r| res[r] * pv[r]).sum();
                        grad[off] += scale * g;
                    }
                    EdgeOperator::FreeBSpline(basis) => {
                        let bmat = precomputed_basis[e_idx].as_ref().unwrap();
                        let g0: f64 = (0..n).map(|r| res[r] * pv[r]).sum();
                        grad[off] += scale * g0;
                        for k in 0..basis.n_basis {
                            let gk: f64 = (0..n).map(|r| res[r] * bmat[r][k]).sum();
                            grad[off + 1 + k] += scale * gk + 2.0 * smooth_lambda * theta[off + 1 + k];
                            mse += smooth_lambda * theta[off + 1 + k].powi(2);
                        }
                    }
                    EdgeOperator::MonotonicISpline { sign, basis } => {
                        let bmat = precomputed_basis[e_idx].as_ref().unwrap();
                        let s0 = sign * spline::sigmoid(theta[off]);
                        let g0: f64 = (0..n).map(|r| res[r] * pv[r]).sum();
                        grad[off] += scale * s0 * g0;
                        for k in 0..basis.n_basis {
                            let ak = theta[off + 1 + k];
                            let wk = spline::softplus(ak);
                            let sk = spline::sigmoid(ak);
                            let gk: f64 = (0..n).map(|r| res[r] * bmat[r][k]).sum();
                            // Smooth L2 penalty บนค่าบวก wk กัน spline โค้งหักศอกที่ปลายหาง
                            grad[off + 1 + k] += scale * sign * sk * gk + 2.0 * smooth_lambda * wk * sk;
                            mse += smooth_lambda * wk * wk;
                        }
                    }
                }
            }

            Ok((mse, grad))
        })?;

        // Centering แต่ละเส้นเชื่อม (เหมือน EBM) และคำนวณ R^2 + Mean Local Elasticity
        let mut centered_intercept = opt.params[0];
        let mut y_hat = vec![opt.params[0]; n];
        for (e_idx, (e, op)) in incoming.iter().zip(ops.into_iter()).enumerate() {
            let off = offsets[e_idx];
            let p_len = op.n_params();
            let e_params = opt.params[off..(off + p_len)].to_vec();
            let pv = parent_vecs[e_idx];

            let mut sum_phi = 0.0_f64;
            let mut sum_deriv = 0.0_f64;
            for r in 0..n {
                let val = op.eval_1d(pv[r], &e_params);
                y_hat[r] += val;
                sum_phi += val;
                sum_deriv += op.deriv_1d(pv[r], &e_params);
            }
            let mean_shift = sum_phi / n_f;
            centered_intercept += mean_shift;

            fitted_edges.push(FittedEdge {
                from: e.from.clone(),
                to: e.to.clone(),
                operator: op,
                params: e_params,
                mean_shift,
                mean_elasticity: sum_deriv / n_f,
            });
        }

        let ss_res: f64 = (0..n).map(|r| (y[r] - y_hat[r]).powi(2)).sum();
        let y_mean = y.iter().sum::<f64>() / n_f;
        let ss_tot: f64 = y.iter().map(|&v| (v - y_mean).powi(2)).sum::<f64>().max(1e-12);

        intercepts.insert(target.clone(), centered_intercept);
        r_squared.insert(target.clone(), 1.0 - ss_res / ss_tot);
    }

    Ok(WhiteBoxModel {
        ir,
        anchors,
        edges: fitted_edges,
        intercepts,
        r_squared,
        calibrators: HashMap::new(),
        interactions: Vec::new(),
        spatial: None,
    })
}

impl WhiteBoxModel {
    /// คืนค่าเส้นโค้ง 1D และ Local Elasticity ของเส้นเชื่อม (White-Box แท้)
    pub fn edge_curve(&self, from: &str, to: &str, n_points: usize) -> Result<EdgeCurve, String> {
        let edge = self
            .edges
            .iter()
            .find(|e| e.from == from && e.to == to)
            .ok_or_else(|| format!("Edge {} -> {} not found", from, to))?;

        let (lo, hi) = match &edge.operator {
            EdgeOperator::FreeBSpline(b) | EdgeOperator::MonotonicISpline { basis: b, .. } => (b.lo, b.hi),
            EdgeOperator::Linear => (-3.0, 3.0),
        };
        let pts = n_points.max(10);
        let mut xs = Vec::with_capacity(pts);
        let mut ys = Vec::with_capacity(pts);
        let mut dys = Vec::with_capacity(pts);

        for i in 0..pts {
            let x = lo + (hi - lo) * (i as f64) / ((pts - 1) as f64);
            xs.push(x);
            ys.push(edge.operator.eval_1d(x, &edge.params) - edge.mean_shift);
            dys.push(edge.operator.deriv_1d(x, &edge.params));
        }

        Ok(EdgeCurve {
            from: from.to_string(),
            to: to.to_string(),
            edge_type: edge.operator.name().to_string(),
            x_grid: xs,
            y_centered: ys,
            local_elasticity: dys,
        })
    }

    /// ทำนายค่า Endogenous Latents บนข้อมูลชุดใหม่
    pub fn predict_latents<T: SemFloat>(
        &self,
        data: MatrixView<'_, T>,
        col_names: &[String],
    ) -> Result<(HashMap<String, Vec<f64>>, HashMap<String, Vec<f64>>), String> {
        let obs_latents = transform_latents(&self.anchors, data, col_names)?;
        let n = data.nrows();
        let mut preds = HashMap::new();

        for target in &self.ir.topo_order {
            let incoming: Vec<_> = self.edges.iter().filter(|e| e.to == *target).collect();
            if incoming.is_empty() {
                continue;
            }
            let b0 = self.intercepts[target];
            let mut y_hat = vec![b0; n];
            for edge in incoming {
                let pv = &obs_latents[&edge.from];
                for r in 0..n {
                    y_hat[r] += edge.operator.eval_1d(pv[r], &edge.params) - edge.mean_shift;
                }
            }
            preds.insert(target.clone(), y_hat);
        }
        Ok((obs_latents, preds))
    }

    /// ปรับเทียบ Split-Conformal Prediction Intervals บนชุด Calibration
    pub fn calibrate_conformal<T: SemFloat>(
        &mut self,
        cal_data: MatrixView<'_, T>,
        col_names: &[String],
        alpha: f64,
    ) -> Result<(), String> {
        let (actual, preds) = self.predict_latents(cal_data, col_names)?;
        for (target, y_hat) in preds {
            let y_true = &actual[&target];
            let residuals: Vec<f64> = y_true.iter().zip(y_hat.iter()).map(|(a, b)| a - b).collect();
            let mut cal = SplitConformalCalibrator::default();
            cal.calibrate(&residuals, alpha);
            self.calibrators.insert(target, cal);
        }
        Ok(())
    }

    /// Structural Intervention do(var += delta) พร้อมแยก Total = Direct + Indirect Effect
    pub fn intervene_shift<T: SemFloat>(
        &self,
        data: MatrixView<'_, T>,
        col_names: &[String],
        intervened_var: &str,
        delta: f64,
        target_var: &str,
    ) -> Result<InterventionSummary, String> {
        let base_latents = transform_latents(&self.anchors, data, col_names)?;
        let n = data.nrows();
        let n_f = n as f64;

        // 1. คำนวณ Residual ประจำแถวของแต่ละ Endogenous Node เพื่อให้ Counterfactual สมบูรณ์
        let mut residuals: HashMap<String, Vec<f64>> = HashMap::new();
        for node in &self.ir.topo_order {
            let incoming: Vec<_> = self.edges.iter().filter(|e| e.to == *node).collect();
            if incoming.is_empty() {
                continue;
            }
            let b0 = self.intercepts[node];
            let mut res = vec![0.0_f64; n];
            for r in 0..n {
                let mut pred = b0;
                for e in &incoming {
                    pred += e.operator.eval_1d(base_latents[&e.from][r], &e.params) - e.mean_shift;
                }
                res[r] = base_latents[node][r] - pred;
            }
            residuals.insert(node.clone(), res);
        }

        // 2. ส่งต่อผลแทรกแซง do(intervened_var += delta) ลงตาม Topological Order (Total Effect)
        let mut cf_latents = base_latents.clone();
        if let Some(v) = cf_latents.get_mut(intervened_var) {
            for x in v.iter_mut() {
                *x += delta;
            }
        } else {
            return Err(format!("Latent '{}' not found", intervened_var));
        }

        for node in &self.ir.topo_order {
            if node == intervened_var {
                continue; // ตัดลูกศรขาเข้า (Graph Mutilation)
            }
            let incoming: Vec<_> = self.edges.iter().filter(|e| e.to == *node).collect();
            if incoming.is_empty() {
                continue;
            }
            let b0 = self.intercepts[node];
            let res = &residuals[node];
            let mut updated = vec![0.0_f64; n];
            for r in 0..n {
                let mut val = b0 + res[r];
                for e in &incoming {
                    val += e.operator.eval_1d(cf_latents[&e.from][r], &e.params) - e.mean_shift;
                }
                updated[r] = val;
            }
            cf_latents.insert(node.clone(), updated);
        }

        let base_t = base_latents
            .get(target_var)
            .ok_or_else(|| format!("Target '{}' not found", target_var))?;
        let cf_t = &cf_latents[target_var];
        let total_effect = (0..n).map(|r| cf_t[r] - base_t[r]).sum::<f64>() / n_f;

        // 3. Direct Effect: เปลี่ยนเฉพาะเส้นเชื่อมตรง intervened_var -> target_var (ตรึงตัวแปรกลางไว้ที่ base)
        let direct_effect = if let Some(dir_edge) = self
            .edges
            .iter()
            .find(|e| e.from == intervened_var && e.to == target_var)
        {
            let x_base = &base_latents[intervened_var];
            (0..n)
                .map(|r| {
                    dir_edge.operator.eval_1d(x_base[r] + delta, &dir_edge.params)
                        - dir_edge.operator.eval_1d(x_base[r], &dir_edge.params)
                })
                .sum::<f64>()
                / n_f
        } else {
            0.0
        };

        let indirect_effect = total_effect - direct_effect;
        let half_w = self
            .calibrators
            .get(target_var)
            .map_or(0.0, |c| c.half_width());

        Ok(InterventionSummary {
            intervened_var: intervened_var.to_string(),
            delta,
            target_var: target_var.to_string(),
            total_effect,
            direct_effect,
            indirect_effect,
            conformal_half_width: half_w,
        })
    }

    /// เทรน ANOVA-Purified 2D Interaction บนข้อมูลเทรนจริงและหักส่วนฉายออกจาก intercept/residuals
    pub fn fit_purified_interaction_with_data<T: SemFloat>(
        &mut self,
        data: MatrixView<'_, T>,
        col_names: &[String],
        var_u: &str,
        var_v: &str,
        target: &str,
        grid_size: usize,
        lr: f64,
        smooth_reg: f64,
        epochs: usize,
    ) -> Result<(), String> {
        let lat_map = transform_latents(&self.anchors, data, col_names)?;
        let zu = lat_map
            .get(var_u)
            .ok_or_else(|| format!("Latent '{}' not found", var_u))?;
        let zv = lat_map
            .get(var_v)
            .ok_or_else(|| format!("Latent '{}' not found", var_v))?;
        let y_target = lat_map
            .get(target)
            .ok_or_else(|| format!("Target latent '{}' not found", target))?;

        let n = data.nrows();
        let b0 = *self.intercepts.get(target).unwrap_or(&0.0);
        let incoming: Vec<_> = self.edges.iter().filter(|e| e.to == target).collect();

        // คำนวณ residual ที่เหลือจาก 1D edge operators ปัจจุบัน
        let mut base_res = vec![0.0_f64; n];
        for r in 0..n {
            let mut pred = b0;
            for edge in &incoming {
                pred += edge.operator.eval_1d(lat_map[&edge.from][r], &edge.params)
                    - edge.mean_shift;
            }
            base_res[r] = pred - y_target[r];
        }

        let mut p2d = Purified2DInteraction::new(
            var_u.to_string(),
            var_v.to_string(),
            zu,
            zv,
            grid_size,
        );

        for _ in 0..epochs {
            let mut current_res = vec![0.0_f64; n];
            for r in 0..n {
                current_res[r] = base_res[r] + p2d.eval_purified(zu[r], zv[r]);
            }
            p2d.step_sgd(zu, zv, &current_res, lr, smooth_reg);
        }

        self.interactions.push(p2d);
        Ok(())
    }

    /// ตั้งค่า Spatial/Network Coupling ด้วย Sparse CSR Matrix (rho = tanh(rho_tilde))
    pub fn set_spatial_coupling(
        &mut self,
        nrows: usize,
        ncols: usize,
        row_ptr: Vec<usize>,
        col_idx: Vec<usize>,
        values: Vec<f64>,
        row_normalize: bool,
        init_rho_tilde: f64,
    ) -> Result<(), String> {
        let csr = CsrMatrix::from_csr(nrows, ncols, row_ptr, col_idx, values)?;
        self.spatial = Some(SpatialCoupling::new(csr, row_normalize, init_rho_tilde));
        Ok(())
    }
}
