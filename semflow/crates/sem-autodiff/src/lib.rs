use faer::prelude::*;
use faer::Side;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeId(pub usize);

#[derive(Clone)]
enum Op {
    Leaf,
    Add(NodeId, NodeId),
    Sub(NodeId, NodeId),
    MatMul(NodeId, NodeId),
    Transpose(NodeId),
    MatInv(NodeId),
    LogDet(NodeId), // Cholesky-based log|Sigma| (คืนค่า 1x1)
    Trace(NodeId),  // tr(X) (คืนค่า 1x1)
    Softplus(NodeId),
    Scale(NodeId, f64),
    SumSq(NodeId), // sum(X_ij^2) สำหรับ ULS / L2 (คืนค่า 1x1)
}

struct Node {
    val: Mat<f64>,
    op: Op,
}

pub struct Tape {
    nodes: Vec<Node>,
}

impl Tape {
    pub fn new() -> Self {
        Self {
            nodes: Vec::with_capacity(64),
        }
    }

    pub fn leaf(&mut self, val: Mat<f64>) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node { val, op: Op::Leaf });
        id
    }

    pub fn value(&self, id: NodeId) -> &Mat<f64> {
        &self.nodes[id.0].val
    }

    pub fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let val = &self.nodes[a.0].val + &self.nodes[b.0].val;
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            val,
            op: Op::Add(a, b),
        });
        id
    }

    pub fn sub(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let val = &self.nodes[a.0].val - &self.nodes[b.0].val;
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            val,
            op: Op::Sub(a, b),
        });
        id
    }

    pub fn matmul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let val = &self.nodes[a.0].val * &self.nodes[b.0].val;
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            val,
            op: Op::MatMul(a, b),
        });
        id
    }

    pub fn transpose(&mut self, a: NodeId) -> NodeId {
        let val = self.nodes[a.0].val.transpose().to_owned();
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            val,
            op: Op::Transpose(a),
        });
        id
    }

    pub fn scale(&mut self, a: NodeId, s: f64) -> NodeId {
        let src = &self.nodes[a.0].val;
        let val = Mat::from_fn(src.nrows(), src.ncols(), |i, j| src[(i, j)] * s);
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            val,
            op: Op::Scale(a, s),
        });
        id
    }

    pub fn mat_inv(&mut self, a: NodeId) -> Result<NodeId, String> {
        let m = &self.nodes[a.0].val;
        let n = m.nrows();
        let lu = m.partial_piv_lu();
        let eye = Mat::<f64>::identity(n, n);
        let inv = lu.solve(&eye);
        if inv.as_ref().norm_max().is_nan() || inv.as_ref().norm_max().is_infinite() {
            return Err("Singular matrix in mat_inv".into());
        }
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            val: inv,
            op: Op::MatInv(a),
        });
        Ok(id)
    }

    /// คำนวณ log|Sigma| ผ่าน Cholesky LLT (รับประกัน Positive Definite)
    pub fn logdet_pd(&mut self, a: NodeId) -> Result<NodeId, String> {
        let m = &self.nodes[a.0].val;
        let llt = m
            .cholesky(Side::Lower)
            .map_err(|_| "Matrix is not positive definite in logdet_pd".to_string())?;
        let l = llt.compute_l();
        let mut logdet = 0.0_f64;
        for i in 0..m.nrows() {
            let d = l[(i, i)];
            if d <= 1e-15 {
                return Err("Near-singular diagonal in Cholesky".into());
            }
            logdet += 2.0 * d.ln();
        }
        let val = Mat::from_fn(1, 1, |_, _| logdet);
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            val,
            op: Op::LogDet(a),
        });
        Ok(id)
    }

    pub fn trace(&mut self, a: NodeId) -> NodeId {
        let m = &self.nodes[a.0].val;
        let mut tr = 0.0_f64;
        for i in 0..m.nrows().min(m.ncols()) {
            tr += m[(i, i)];
        }
        let val = Mat::from_fn(1, 1, |_, _| tr);
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            val,
            op: Op::Trace(a),
        });
        id
    }

    pub fn softplus(&mut self, a: NodeId) -> NodeId {
        let m = &self.nodes[a.0].val;
        let val = Mat::from_fn(m.nrows(), m.ncols(), |i, j| {
            let x = m[(i, j)];
            if x > 20.0 {
                x
            } else {
                (1.0 + x.exp()).ln()
            }
        });
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            val,
            op: Op::Softplus(a),
        });
        id
    }

    pub fn sum_sq(&mut self, a: NodeId) -> NodeId {
        let m = &self.nodes[a.0].val;
        let mut s = 0.0_f64;
        for i in 0..m.nrows() {
            for j in 0..m.ncols() {
                s += m[(i, j)] * m[(i, j)];
            }
        }
        let val = Mat::from_fn(1, 1, |_, _| s);
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            val,
            op: Op::SumSq(a),
        });
        id
    }

    /// Reverse-Mode Backpropagation จาก Scalar Loss Node
    pub fn backward(&self, loss_id: NodeId) -> Vec<Mat<f64>> {
        let mut grads: Vec<Mat<f64>> = self
            .nodes
            .iter()
            .map(|n| Mat::zeros(n.val.nrows(), n.val.ncols()))
            .collect();
        grads[loss_id.0][(0, 0)] = 1.0;

        for idx in (0..=loss_id.0).rev() {
            let g_out = grads[idx].clone();
            match self.nodes[idx].op {
                Op::Leaf => {}
                Op::Add(a, b) => {
                    grads[a.0] = &grads[a.0] + &g_out;
                    grads[b.0] = &grads[b.0] + &g_out;
                }
                Op::Sub(a, b) => {
                    grads[a.0] = &grads[a.0] + &g_out;
                    grads[b.0] = &grads[b.0] - &g_out;
                }
                Op::MatMul(a, b) => {
                    let va = &self.nodes[a.0].val;
                    let vb = &self.nodes[b.0].val;
                    let ga = &g_out * vb.transpose();
                    let gb = va.transpose() * &g_out;
                    grads[a.0] = &grads[a.0] + &ga;
                    grads[b.0] = &grads[b.0] + &gb;
                }
                Op::Transpose(a) => {
                    grads[a.0] = &grads[a.0] + g_out.transpose();
                }
                Op::Scale(a, s) => {
                    let ga = Mat::from_fn(g_out.nrows(), g_out.ncols(), |i, j| g_out[(i, j)] * s);
                    grads[a.0] = &grads[a.0] + &ga;
                }
                Op::MatInv(a) => {
                    // dL/dX = -Y^T * g_out * Y^T โดย Y = X^{-1}
                    let y = &self.nodes[idx].val;
                    let yt = y.transpose();
                    let tmp = yt * &g_out * yt;
                    grads[a.0] = &grads[a.0] - &tmp;
                }
                Op::LogDet(a) => {
                    // d(log|Sigma|)/dSigma = (Sigma^{-1})^T
                    let sigma = &self.nodes[a.0].val;
                    let s = g_out[(0, 0)];
                    if let Ok(llt) = sigma.cholesky(Side::Lower) {
                        let inv = llt.solve(&Mat::<f64>::identity(sigma.nrows(), sigma.ncols()));
                        let scaled =
                            Mat::from_fn(inv.nrows(), inv.ncols(), |i, j| inv[(j, i)] * s);
                        grads[a.0] = &grads[a.0] + &scaled;
                    }
                }
                Op::Trace(a) => {
                    let s = g_out[(0, 0)];
                    let n = grads[a.0].nrows().min(grads[a.0].ncols());
                    for i in 0..n {
                        grads[a.0][(i, i)] += s;
                    }
                }
                Op::Softplus(a) => {
                    let x = &self.nodes[a.0].val;
                    for i in 0..x.nrows() {
                        for j in 0..x.ncols() {
                            let sig = 1.0 / (1.0 + (-x[(i, j)]).exp());
                            grads[a.0][(i, j)] += g_out[(i, j)] * sig;
                        }
                    }
                }
                Op::SumSq(a) => {
                    let s = g_out[(0, 0)];
                    let x = &self.nodes[a.0].val;
                    for i in 0..x.nrows() {
                        for j in 0..x.ncols() {
                            grads[a.0][(i, j)] += 2.0 * s * x[(i, j)];
                        }
                    }
                }
            }
        }
        grads
    }
}

/// Two-Loop L-BFGS Optimizer พร้อม Armijo Backtracking Line Search
pub struct LbfgsResult {
    pub params: Vec<f64>,
    pub loss: f64,
    pub iterations: usize,
    pub converged: bool,
}

pub fn minimize_lbfgs<F>(
    init: &[f64],
    max_iter: usize,
    tol: f64,
    m_history: usize,
    mut eval_fn: F,
) -> Result<LbfgsResult, String>
where
    F: FnMut(&[f64]) -> Result<(f64, Vec<f64>), String>,
{
    let d = init.len();
    let mut x = init.to_vec();
    let (mut f, mut g) = eval_fn(&x)?;

    let mut s_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(m_history);
    let mut y_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(m_history);
    let mut rho_hist: VecDeque<f64> = VecDeque::with_capacity(m_history);

    let dot = |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).map(|(u, v)| u * v).sum() };

    for iter in 1..=max_iter {
        let g_norm = dot(&g, &g).sqrt();
        if g_norm < tol {
            return Ok(LbfgsResult {
                params: x,
                loss: f,
                iterations: iter - 1,
                converged: true,
            });
        }

        // Two-loop recursion หาทิศทางค้นหา p = -H_k * g_k
        let mut q = g.clone();
        let k = s_hist.len();
        let mut alpha = vec![0.0_f64; k];
        for i in (0..k).rev() {
            alpha[i] = rho_hist[i] * dot(&s_hist[i], &q);
            for j in 0..d {
                q[j] -= alpha[i] * y_hist[i][j];
            }
        }
        let gamma = if k > 0 {
            let yy = dot(&y_hist[k - 1], &y_hist[k - 1]).max(1e-16);
            dot(&s_hist[k - 1], &y_hist[k - 1]) / yy
        } else {
            1.0
        };
        let mut r: Vec<f64> = q.iter().map(|&v| gamma * v).collect();
        for i in 0..k {
            let beta = rho_hist[i] * dot(&y_hist[i], &r);
            for j in 0..d {
                r[j] += s_hist[i][j] * (alpha[i] - beta);
            }
        }
        let mut p: Vec<f64> = r.iter().map(|&v| -v).collect();
        let mut dir_deriv = dot(&g, &p);
        if dir_deriv >= 0.0 {
            // Reset เป็น Steepest Descent หากทิศทางไม่ลดค่า
            p = g.iter().map(|&v| -v).collect();
            dir_deriv = dot(&g, &p);
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
        }

        // Armijo Backtracking Line Search (กันเมทริกซ์หลุด Positive Definite ด้วย)
        let mut step = 1.0_f64;
        let mut accepted = None;
        for _ in 0..25 {
            let x_new: Vec<f64> = (0..d).map(|j| x[j] + step * p[j]).collect();
            if let Ok((f_new, g_new)) = eval_fn(&x_new) {
                if f_new <= f + 1e-4 * step * dir_deriv {
                    accepted = Some((x_new, f_new, g_new));
                    break;
                }
            }
            step *= 0.5;
        }

        let Some((x_new, f_new, g_new)) = accepted else {
            return Ok(LbfgsResult {
                params: x,
                loss: f,
                iterations: iter,
                converged: g_norm < tol * 10.0,
            });
        };

        let s: Vec<f64> = (0..d).map(|j| x_new[j] - x[j]).collect();
        let y: Vec<f64> = (0..d).map(|j| g_new[j] - g[j]).collect();
        let sy = dot(&s, &y);
        if sy > 1e-12 {
            if s_hist.len() == m_history {
                s_hist.pop_front();
                y_hist.pop_front();
                rho_hist.pop_front();
            }
            s_hist.push_back(s);
            y_hist.push_back(y);
            rho_hist.push_back(1.0 / sy);
        }

        let f_diff = (f - f_new).abs();
        x = x_new;
        f = f_new;
        g = g_new;

        if f_diff < 1e-14 && g_norm < tol * 10.0 {
            return Ok(LbfgsResult {
                params: x,
                loss: f,
                iterations: iter,
                converged: true,
            });
        }
    }

    Ok(LbfgsResult {
        params: x,
        loss: f,
        iterations: max_iter,
        converged: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finite_difference_gradient_ml_loss() {
        // ตรวจ Gradient ของ log|Sigma| + tr(S * Sigma^{-1}) เทียบ Finite Differences
        let s_obs = Mat::from_fn(2, 2, |i, j| if i == j { 1.5 } else { 0.4 });
        let sigma0 = Mat::from_fn(2, 2, |i, j| if i == j { 2.0 } else { 0.3 });

        let eval = |sig: &Mat<f64>| -> (f64, Mat<f64>) {
            let mut tape = Tape::new();
            let s_id = tape.leaf(s_obs.clone());
            let sig_id = tape.leaf(sig.clone());
            let ld = tape.logdet_pd(sig_id).unwrap();
            let inv = tape.mat_inv(sig_id).unwrap();
            let prod = tape.matmul(s_id, inv);
            let tr = tape.trace(prod);
            let loss = tape.add(ld, tr);
            let grads = tape.backward(loss);
            (tape.value(loss)[(0, 0)], grads[sig_id.0].clone())
        };

        let (_, grad) = eval(&sigma0);
        let eps = 1e-6;
        for i in 0..2 {
            for j in i..2 {
                let mut sp = sigma0.clone();
                let mut sm = sigma0.clone();
                sp[(i, j)] += eps;
                sm[(i, j)] -= eps;
                if i != j {
                    sp[(j, i)] += eps;
                    sm[(j, i)] -= eps;
                }
                let num_g = (eval(&sp).0 - eval(&sm).0) / (2.0 * eps);
                let ana_g = if i == j {
                    grad[(i, j)]
                } else {
                    grad[(i, j)] + grad[(j, i)]
                };
                assert!(
                    (ana_g - num_g).abs() < 1e-6,
                    "Grad mismatch at ({},{}): ana={} num={}",
                    i,
                    j,
                    ana_g,
                    num_g
                );
            }
        }
    }
}
