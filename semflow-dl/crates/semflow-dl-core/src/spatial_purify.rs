use crate::spline::{sigmoid, softplus};
use rayon::prelude::*;

/// Sparse Matrix แบบ Compressed Sparse Row (CSR) ประหยัด RAM สำหรับเมทริกซ์ W ขนาดใหญ่
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    pub nrows: usize,
    pub ncols: usize,
    pub row_ptr: Vec<usize>, // ความยาว nrows + 1
    pub col_idx: Vec<usize>, // ความยาว nnz
    pub values: Vec<f64>,  // ความยาว nnz
}

impl CsrMatrix {
    pub fn from_csr(
        nrows: usize,
        ncols: usize,
        row_ptr: Vec<usize>,
        col_idx: Vec<usize>,
        values: Vec<f64>,
    ) -> Result<Self, String> {
        if row_ptr.len() != nrows + 1 {
            return Err("Invalid CSR row_ptr length".into());
        }
        let nnz = row_ptr[nrows];
        if col_idx.len() != nnz || values.len() != nnz {
            return Err("CSR col_idx/values length mismatch with row_ptr".into());
        }
        for &c in &col_idx {
            if c >= ncols {
                return Err(format!("CSR column index {} out of bounds ({})", c, ncols));
            }
        }
        Ok(Self {
            nrows,
            ncols,
            row_ptr,
            col_idx,
            values,
        })
    }

    /// Row-normalize แต่ละแถวที่มีผลรวมมากกว่า zero ให้ผลรวมของแถวเท่ากับ 1.0 เพื่อเสถียรภาพของ rho * W
    pub fn row_normalize(&mut self) {
        for r in 0..self.nrows {
            let start = self.row_ptr[r];
            let end = self.row_ptr[r + 1];
            let row_sum: f64 = self.values[start..end].iter().map(|v| v.abs()).sum();
            if row_sum > 1e-12 {
                for v in &mut self.values[start..end] {
                    *v /= row_sum;
                }
            }
        }
    }

    /// คูณเมทริกซ์ CSR กับเวกเตอร์ y = W * x แบบขนานด้วย rayon
    pub fn matvec(&self, x: &[f64]) -> Result<Vec<f64>, String> {
        if x.len() != self.ncols {
            return Err(format!("CSR matvec dim mismatch: got {}, expected {}", x.len(), self.ncols));
        }
        let out: Vec<f64> = (0..self.nrows)
            .into_par_iter()
            .map(|r| {
                let start = self.row_ptr[r];
                let end = self.row_ptr[r + 1];
                let mut acc = 0.0_f64;
                for idx in start..end {
                    acc += self.values[idx] * x[self.col_idx[idx]];
                }
                acc
            })
            .collect();
        Ok(out)
    }

    /// คูณทรานสโพสของ CSR กับเวกเตอร์ g = W^T * v (ใช้สำหรับ Backpropagation ของ Spatial Coupling)
    pub fn matvec_transpose(&self, v: &[f64]) -> Result<Vec<f64>, String> {
        if v.len() != self.nrows {
            return Err(format!(
                "CSR matvec_transpose dim mismatch: got {}, expected {}",
                v.len(),
                self.nrows
            ));
        }
        let mut out = vec![0.0_f64; self.ncols];
        for r in 0..self.nrows {
            let vr = v[r];
            if vr == 0.0 {
                continue;
            }
            let start = self.row_ptr[r];
            let end = self.row_ptr[r + 1];
            for idx in start..end {
                out[self.col_idx[idx]] += self.values[idx] * vr;
            }
        }
        Ok(out)
    }
}

/// ตัวเชื่อม Spatial/Network Coupling: y_spatial = y_dag + rho * (W * y_dag)
/// โดย rho = tanh(rho_tilde) อยู่ในช่วง (-1, 1) เสมอ
#[derive(Debug, Clone)]
pub struct SpatialCoupling {
    pub w_csr: CsrMatrix,
    pub rho_tilde: f64,
}

impl SpatialCoupling {
    pub fn new(mut w_csr: CsrMatrix, row_norm: bool, init_rho_tilde: f64) -> Self {
        if row_norm {
            w_csr.row_normalize();
        }
        Self {
            w_csr,
            rho_tilde: init_rho_tilde,
        }
    }

    #[inline]
    pub fn rho(&self) -> f64 {
        self.rho_tilde.tanh()
    }

    #[inline]
    pub fn drho_drho_tilde(&self) -> f64 {
        let t = self.rho_tilde.tanh();
        1.0 - t * t
    }

    /// Forward: y_out = y_dag + rho * (W * y_dag)
    pub fn forward(&self, y_dag: &[f64]) -> Result<(Vec<f64>, Vec<f64>), String> {
        let wy = self.w_csr.matvec(y_dag)?;
        let rho = self.rho();
        let y_out: Vec<f64> = y_dag
            .iter()
            .zip(wy.iter())
            .map(|(&yd, &w_val)| yd + rho * w_val)
            .collect();
        Ok((y_out, wy))
    }

    /// Backward: คืนค่า (grad_y_dag, grad_rho_tilde) จาก upstream gradient grad_y_out
    pub fn backward(
        &self,
        grad_y_out: &[f64],
        wy_cached: &[f64],
    ) -> Result<(Vec<f64>, f64), String> {
        let rho = self.rho();
        let wt_g = self.w_csr.matvec_transpose(grad_y_out)?;
        let grad_y_dag: Vec<f64> = grad_y_out
            .iter()
            .zip(wt_g.iter())
            .map(|(&go, &wg)| go + rho * wg)
            .collect();

        let d_rho: f64 = grad_y_out
            .iter()
            .zip(wy_cached.iter())
            .map(|(&go, &w_val)| go * w_val)
            .sum();
        let grad_rho_tilde = d_rho * self.drho_drho_tilde();
        Ok((grad_y_dag, grad_rho_tilde))
    }
}

/// ANOVA-Purified 2D Interaction Basis บนคู่ตัวแปร (u, v)
/// หักผลเฉลี่ยตามแกน u และแกน v ออก เพื่อให้ psi_uv แสดงเฉพาะ Pure 2D Interaction
#[derive(Debug, Clone)]
pub struct Purified2DInteraction {
    pub var_u: String,
    pub var_v: String,
    pub grid_u: Vec<f64>, // ความยาว G_u
    pub grid_v: Vec<f64>, // ความยาว G_v
    pub bw_u: f64,
    pub bw_v: f64,
    /// เมทริกซ์น้ำหนักขนาด G_u x G_v (เก็บแบบ row-major ความยาว G_u * G_v)
    pub weights: Vec<f64>,
    /// ค่าปรับ purifications ตาม marginal ของ u และ v เพื่อรับประกันค่าเฉลี่ยเป็นศูนย์ทั้งสองแกน
    pub row_bias: Vec<f64>, // ความยาว G_u
    pub col_bias: Vec<f64>, // ความยาว G_v
    pub grand_bias: f64,
}

impl Purified2DInteraction {
    pub fn new(
        var_u: String,
        var_v: String,
        vals_u: &[f64],
        vals_v: &[f64],
        grid_size: usize,
    ) -> Self {
        let g = grid_size.clamp(3, 12);
        let make_grid = |arr: &[f64]| -> (Vec<f64>, f64) {
            let mut s: Vec<f64> = arr.iter().copied().filter(|x| x.is_finite()).collect();
            s.sort_by(|a, b| a.total_cmp(b));
            let lo = s.first().copied().unwrap_or(-2.5);
            let hi = s.last().copied().unwrap_or(2.5);
            let span = if (hi - lo).abs() < 1e-6 { 5.0 } else { hi - lo };
            let grid: Vec<f64> = (0..g)
                .map(|i| lo + span * (i as f64) / ((g - 1) as f64))
                .collect();
            let bw = (span / ((g - 1) as f64)).max(0.15);
            (grid, bw)
        };

        let (grid_u, bw_u) = make_grid(vals_u);
        let (grid_v, bw_v) = make_grid(vals_v);

        let mut p = Self {
            var_u,
            var_v,
            grid_u,
            grid_v,
            bw_u,
            bw_v,
            weights: vec![0.0_f64; g * g],
            row_bias: vec![0.0_f64; g],
            col_bias: vec![0.0_f64; g],
            grand_bias: 0.0,
        };
        p.recompute_anova_purification();
        p
    }

    #[inline]
    fn rbf_weights(x: f64, grid: &[f64], bw: f64) -> Vec<f64> {
        let inv_2s2 = -0.5 / (bw * bw);
        let mut raw: Vec<f64> = grid.iter().map(|&gk| ((x - gk).powi(2) * inv_2s2).exp()).collect();
        let sum: f64 = raw.iter().sum();
        if sum > 1e-15 {
            for w in &mut raw {
                *w /= sum;
            }
        } else {
            let uniform = 1.0 / (grid.len() as f64);
            raw.fill(uniform);
        }
        raw
    }

    /// คำนวณ Row/Column/Grand Purification Terms เพื่อบังคับ ANOVA orthogonality
    pub fn recompute_anova_purification(&mut self) {
        let gu = self.grid_u.len();
        let gv = self.grid_v.len();
        let gvf = gv as f64;
        let guf = gu as f64;

        self.row_bias = (0..gu)
            .map(|a| {
                let s: f64 = (0..gv).map(|b| self.weights[a * gv + b]).sum();
                s / gvf
            })
            .collect();

        self.col_bias = (0..gv)
            .map(|b| {
                let s: f64 = (0..gu).map(|a| self.weights[a * gv + b]).sum();
                s / guf
            })
            .collect();

        self.grand_bias = self.weights.iter().sum::<f64>() / (guf * gvf);
    }

    /// ประเมินค่า ANOVA-Purified 2D Interaction ณ จุด (zu, zv)
    pub fn eval_purified(&self, zu: f64, zv: f64) -> f64 {
        let wu = Self::rbf_weights(zu, &self.grid_u, self.bw_u);
        let wv = Self::rbf_weights(zv, &self.grid_v, self.bw_v);
        let gu = self.grid_u.len();
        let gv = self.grid_v.len();

        let mut total = 0.0_f64;
        let mut marg_u = 0.0_f64;
        let mut marg_v = 0.0_f64;

        for (a, &ua) in wu.iter().enumerate() {
            let mut row_acc = 0.0_f64;
            for (b, &vb) in wv.iter().enumerate() {
                let w_ab = self.weights[a * gv + b];
                total += ua * vb * w_ab;
                row_acc += w_ab;
            }
            marg_u += ua * (row_acc / (gv as f64));
        }

        for (b, &vb) in wv.iter().enumerate() {
            let col_acc: f64 = (0..gu).map(|a| self.weights[a * gv + b]).sum();
            marg_v += vb * (col_acc / (gu as f64));
        }

        let mean_all = self.weights.iter().sum::<f64>() / ((gu * gv) as f64);
        total - marg_u - marg_v + mean_all
    }

    /// อัปเดตน้ำหนักแบบ Gradient Descent พร้อม Smoothness L2 Penalty
    pub fn step_sgd(
        &mut self,
        vals_u: &[f64],
        vals_v: &[f64],
        residuals: &[f64], // pred - target
        lr: f64,
        smooth_reg: f64,
    ) {
        let n = residuals.len();
        if n == 0 {
            return;
        }
        let gu = self.grid_u.len();
        let gv = self.grid_v.len();
        let mut grad_w = vec![0.0_f64; gu * gv];
        let scale = 2.0 / (n as f64);

        for r in 0..n {
            let err = residuals[r];
            if err == 0.0 || !err.is_finite() {
                continue;
            }
            let wu = Self::rbf_weights(vals_u[r], &self.grid_u, self.bw_u);
            let wv = Self::rbf_weights(vals_v[r], &self.grid_v, self.bw_v);
            let g_coef = scale * err;
            for (a, &ua) in wu.iter().enumerate() {
                for (b, &vb) in wv.iter().enumerate() {
                    grad_w[a * gv + b] += g_coef * ua * vb;
                }
            }
        }

        for idx in 0..(gu * gv) {
            let g_tot = grad_w[idx] + 2.0 * smooth_reg * self.weights[idx];
            self.weights[idx] -= lr * g_tot;
        }
        self.recompute_anova_purification();
    }

    /// ช่วยคำนวณค่า softplus / sigmoid ในกรณีต้องการพารามิเตอร์บวกเสริม
    pub fn positive_gate(raw: f64) -> (f64, f64) {
        (softplus(raw), sigmoid(raw))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_spatial_stability_bound() {
        // กราฟวงแหวน 3 โหนด (0 -> 1 -> 2 -> 0)
        let csr = CsrMatrix::from_csr(
            3,
            3,
            vec![0, 1, 2, 3],
            vec![1, 2, 0],
            vec![2.0, 4.0, 0.5],
        )
        .unwrap();
        let mut sp = SpatialCoupling::new(csr, true, 5.0);
        // แม้ rho_tilde = 5.0 แต่ tanh(5.0) < 1.0 และไม่เกิด NaN/Inf
        assert!(sp.rho() > 0.999 && sp.rho() < 1.0);

        let y_dag = vec![1.0, -0.5, 0.25];
        let (y_out, wy) = sp.forward(&y_dag).unwrap();
        assert_eq!(y_out.len(), 3);

        let (g_ydag, g_rho_tilde) = sp.backward(&vec![1.0, 1.0, 1.0], &wy).unwrap();
        assert_eq!(g_ydag.len(), 3);
        assert!(g_rho_tilde.is_finite());
    }

    #[test]
    fn test_anova_purified_2d_zero_marginal() {
        let u: Vec<f64> = (-50..=50).map(|i| (i as f64) * 0.04).collect();
        let v: Vec<f64> = (-50..=50).map(|i| ((i * 7) % 101 - 50) as f64 * 0.04).collect();
        let mut p2d = Purified2DInteraction::new("U".into(), "V".into(), &u, &v, 5);
        p2d.weights = (0..25).map(|i| ((i as f64) - 12.0) * 0.05).collect();
        p2d.recompute_anova_purification();

        let mean_over_u: f64 = p2d.grid_u.iter().map(|&gu| p2d.eval_purified(gu, 0.0)).sum::<f64>()
            / (p2d.grid_u.len() as f64);
        let mean_over_v: f64 = p2d.grid_v.iter().map(|&gv| p2d.eval_purified(0.0, gv)).sum::<f64>()
            / (p2d.grid_v.len() as f64);
        assert!(mean_over_u.abs() < 0.15);
        assert!(mean_over_v.abs() < 0.15);
    }
}
