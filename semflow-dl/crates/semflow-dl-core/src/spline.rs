use sem_traits::StructuralOperator;

#[inline]
pub fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[inline]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// ฐาน Spline ลำดับ k (ค่าเริ่มต้น k = 3: Cubic I-Spline เรียบระดับ C^2, Quadratic M-Spline)
#[derive(Debug, Clone)]
pub struct SplineBasis {
    pub order: usize,
    pub lo: f64,
    pub hi: f64,
    /// Extended clamped knot vector ความยาว = n_interior + 2*(order + 1)
    pub knots: Vec<f64>,
    pub n_basis: usize,
}

impl SplineBasis {
    /// สร้างปม (Knots) ตามควอนไทล์ของข้อมูลจริง เพื่อไม่ให้ฐานว่างเปล่าที่ปลายหาง
    pub fn from_quantiles(values: &[f64], n_interior: usize, order: usize) -> Self {
        let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let lo = sorted.first().copied().unwrap_or(-3.0) - 0.5;
        let hi = sorted.last().copied().unwrap_or(3.0) + 0.5;
        let (lo, hi) = if (hi - lo).abs() < 1e-8 {
            (lo - 1.0, hi + 1.0)
        } else {
            (lo, hi)
        };

        let mut interior = Vec::with_capacity(n_interior);
        for q in 1..=n_interior {
            let frac = (q as f64) / ((n_interior + 1) as f64);
            let idx = ((sorted.len().saturating_sub(1)) as f64 * frac).round() as usize;
            let k_val = sorted.get(idx).copied().unwrap_or(lo + frac * (hi - lo));
            if k_val > lo + 1e-6 && k_val < hi - 1e-6 {
                if interior.last().map_or(true, |&prev| k_val > prev + 1e-6) {
                    interior.push(k_val);
                }
            }
        }
        Self::new(lo, hi, &interior, order)
    }

    pub fn new(lo: f64, hi: f64, interior: &[f64], order: usize) -> Self {
        let k = order.max(2);
        let mut knots = Vec::with_capacity(interior.len() + 2 * (k + 1));
        for _ in 0..=k {
            knots.push(lo);
        }
        knots.extend_from_slice(interior);
        for _ in 0..=k {
            knots.push(hi);
        }
        let n_basis = interior.len() + k;
        Self {
            order: k,
            lo,
            hi,
            knots,
            n_basis,
        }
    }

    /// คืนค่า (I_splines[0..B], M_splines[0..B]) ณ จุด x โดย I'_b(x) == M_b(x) เป๊ะ
    pub fn eval_ispline_and_mspline(&self, x: f64) -> (Vec<f64>, Vec<f64>) {
        let b_cnt = self.n_basis;
        let k = self.order;
        let mut i_out = vec![0.0_f64; b_cnt];
        let mut m_out = vec![0.0_f64; b_cnt];

        if x <= self.lo {
            return (i_out, m_out);
        }
        if x >= self.hi {
            i_out.fill(1.0);
            return (i_out, m_out);
        }

        let n_t = self.knots.len();
        // หาช่วง j ที่ t[j] <= x < t[j+1] (อยู่ระหว่าง k ..= k + n_interior)
        let mut j = k;
        while j + 1 < n_t - (k + 1) && x >= self.knots[j + 1] {
            j += 1;
        }

        // Cox-de Boor / Ramsay recursion: คำนวณ M_{m, r}(x) ตั้งแต่ r = 1 ถึง k + 1
        let mut m_prev = vec![0.0_f64; n_t];
        let denom1 = self.knots[j + 1] - self.knots[j];
        if denom1 > 1e-15 {
            m_prev[j] = 1.0 / denom1;
        }

        let mut m_order_k = vec![0.0_f64; n_t];
        for r in 2..=(k + 1) {
            let mut m_curr = vec![0.0_f64; n_t];
            let rf = r as f64;
            let r_minus_1 = (r - 1) as f64;
            for idx in 0..(n_t - r) {
                let span = self.knots[idx + r] - self.knots[idx];
                if span > 1e-15 {
                    let left = (x - self.knots[idx]) * m_prev[idx];
                    let right = (self.knots[idx + r] - x) * m_prev[idx + 1];
                    m_curr[idx] = rf * (left + right) / (r_minus_1 * span);
                }
            }
            if r == k {
                m_order_k = m_curr.clone();
            }
            m_prev = m_curr;
        }
        // ตอนนี้ m_prev คือ M_{m, k+1}(x) และ m_order_k คือ M_{m, k}(x)

        let kp1 = (k + 1) as f64;
        for b in 0..b_cnt {
            let i = b + 1; // ดัชนีใน extended knot vector
            m_out[b] = m_order_k[i];

            if i > j {
                i_out[b] = 0.0;
            } else if i + k <= j {
                i_out[b] = 1.0;
            } else {
                let mut acc = 0.0_f64;
                for m in i..=j {
                    let span = self.knots[m + k + 1] - self.knots[m];
                    acc += (span / kp1) * m_prev[m];
                }
                i_out[b] = acc.clamp(0.0, 1.0);
            }
        }

        (i_out, m_out)
    }

    /// ฐาน B-Spline อิสระ (ใช้ M_{i,k}(x) ปรับสเกลด้วยความกว้างช่วงให้รวมกันได้ 1)
    pub fn eval_bspline_and_deriv(&self, x: f64) -> (Vec<f64>, Vec<f64>) {
        // ใช้ I-Spline เป็นฐานที่เรียบและหาอนุพันธ์ตรงด้วย M-Spline (ปล่อยน้ำหนักเป็นบวกหรือลบได้อิสระ)
        self.eval_ispline_and_mspline(x)
    }
}

/// Edge Operator ที่ implement sem_traits::StructuralOperator โดยตรง
#[derive(Debug, Clone)]
pub enum EdgeOperator {
    /// phi(z) = beta * z  (1 param)
    Linear,
    /// phi(z) = w0 * z + sum_b c_b * I_b(z)  (1 + B params, อิสระจับ U-shape ได้)
    FreeBSpline(SplineBasis),
    /// phi(z) = sign * (softplus(a0)*z + sum_b softplus(a_{b+1})*I_b(z))
    /// รับประกัน Monotonicity 100% ทุกจุดบนแกนจำนวนจริง และมีอนุพันธ์แม่นยำ phi'(z)
    MonotonicISpline { sign: f64, basis: SplineBasis },
}

impl EdgeOperator {
    pub fn n_params(&self) -> usize {
        match self {
            Self::Linear => 1,
            Self::FreeBSpline(b) => 1 + b.n_basis,
            Self::MonotonicISpline { basis, .. } => 1 + basis.n_basis,
        }
    }

    pub fn init_params(&self) -> Vec<f64> {
        match self {
            Self::Linear => vec![0.2],
            Self::FreeBSpline(b) => {
                let mut v = vec![0.0; 1 + b.n_basis];
                v[0] = 0.2;
                v
            }
            Self::MonotonicISpline { basis, .. } => {
                // softplus(-1.5) ≈ 0.20 ให้เริ่มใกล้เชิงเส้นเรียบๆ
                vec![-1.5; 1 + basis.n_basis]
            }
        }
    }

    /// คืนค่า (phi(x), dphi/dparams) สำหรับการเทรนด้วย L-BFGS
    pub fn eval_and_param_grad(&self, x: f64, params: &[f64]) -> (f64, Vec<f64>) {
        match self {
            Self::Linear => (params[0] * x, vec![x]),
            Self::FreeBSpline(basis) => {
                let (i_vals, _) = basis.eval_bspline_and_deriv(x);
                let mut val = params[0] * x;
                let mut g = Vec::with_capacity(1 + basis.n_basis);
                g.push(x);
                for (b, &ib) in i_vals.iter().enumerate() {
                    val += params[b + 1] * ib;
                    g.push(ib);
                }
                (val, g)
            }
            Self::MonotonicISpline { sign, basis } => {
                let (i_vals, _) = basis.eval_ispline_and_mspline(x);
                let w0 = softplus(params[0]);
                let mut val = w0 * x;
                let mut g = Vec::with_capacity(1 + basis.n_basis);
                g.push(sign * sigmoid(params[0]) * x);
                for (b, &ib) in i_vals.iter().enumerate() {
                    let wb = softplus(params[b + 1]);
                    val += wb * ib;
                    g.push(sign * sigmoid(params[b + 1]) * ib);
                }
                (sign * val, g)
            }
        }
    }
}

impl StructuralOperator for EdgeOperator {
    fn name(&self) -> &str {
        match self {
            Self::Linear => "linear",
            Self::FreeBSpline(_) => "bspline",
            Self::MonotonicISpline { .. } => "ispline",
        }
    }

    fn eval_1d(&self, x: f64, params: &[f64]) -> f64 {
        match self {
            Self::Linear => params[0] * x,
            Self::FreeBSpline(basis) => {
                let (i_vals, _) = basis.eval_bspline_and_deriv(x);
                let mut val = params[0] * x;
                for (b, &ib) in i_vals.iter().enumerate() {
                    val += params[b + 1] * ib;
                }
                val
            }
            Self::MonotonicISpline { sign, basis } => {
                let (i_vals, _) = basis.eval_ispline_and_mspline(x);
                let mut val = softplus(params[0]) * x;
                for (b, &ib) in i_vals.iter().enumerate() {
                    val += softplus(params[b + 1]) * ib;
                }
                sign * val
            }
        }
    }

    fn deriv_1d(&self, x: f64, params: &[f64]) -> f64 {
        match self {
            Self::Linear => params[0],
            Self::FreeBSpline(basis) => {
                let (_, m_vals) = basis.eval_bspline_and_deriv(x);
                let mut d = params[0];
                for (b, &mb) in m_vals.iter().enumerate() {
                    d += params[b + 1] * mb;
                }
                d
            }
            Self::MonotonicISpline { sign, basis } => {
                let (_, m_vals) = basis.eval_ispline_and_mspline(x);
                let mut d = softplus(params[0]);
                for (b, &mb) in m_vals.iter().enumerate() {
                    d += softplus(params[b + 1]) * mb;
                }
                sign * d
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ispline_derivative_and_monotonicity() {
        let basis = SplineBasis::new(-2.0, 2.0, &[-1.0, 0.0, 1.0], 3);
        let op = EdgeOperator::MonotonicISpline {
            sign: 1.0,
            basis: basis.clone(),
        };
        let params = vec![-1.2, 0.5, -0.3, 1.1, 0.2, -0.7, 0.8];
        let eps = 1e-6;

        let mut prev_y = f64::NEG_INFINITY;
        for step in 0..=80 {
            let x = -2.2 + 4.4 * (step as f64) / 80.0;
            let y = op.eval_1d(x, &params);
            let dy = op.deriv_1d(x, &params);

            // 1. การันตี Monotonic เพิ่มขึ้นเสมอทุกจุด (รวมถึงนอกช่วง knots)
            assert!(y >= prev_y - 1e-12, "Monotonicity violated at x={}", x);
            assert!(dy > 0.0, "Derivative must be strictly positive at x={}", x);
            prev_y = y;

            // 2. ตรวจอนุพันธ์ M-Spline เทียบ Central Finite Differences (หลบจุดขอบ knot เป๊ะ)
            if (x - (-2.0)).abs() > 1e-3 && (x - 2.0).abs() > 1e-3 {
                let num_dy = (op.eval_1d(x + eps, &params) - op.eval_1d(x - eps, &params)) / (2.0 * eps);
                assert!(
                    (dy - num_dy).abs() < 1e-6,
                    "Derivative mismatch at x={}: exact={}, num={}",
                    x,
                    dy,
                    num_dy
                );
            }
        }
    }
}
