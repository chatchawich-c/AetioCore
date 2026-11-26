use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use sem_traits::{Calibrator, MatrixLayout, MatrixView, StructuralOperator};
use semflow_dl_core::{fit_whitebox_sem, WhiteBoxModel};
use std::collections::HashMap;

#[pyclass]
struct PyWhiteBoxModel {
    inner: WhiteBoxModel,
}

#[pymethods]
impl PyWhiteBoxModel {
    #[getter]
    fn r_squared(&self) -> HashMap<String, f64> {
        self.inner.r_squared.clone()
    }

    #[getter]
    fn edges_summary(&self) -> Vec<(String, String, String, f64)> {
        self.inner
            .edges
            .iter()
            .map(|e| {
                (
                    e.from.clone(),
                    e.to.clone(),
                    e.operator.name().to_string(),
                    e.mean_elasticity,
                )
            })
            .collect()
    }

    #[pyo3(signature = (from_var, to_var, n_points=100))]
    fn edge_curve(
        &self,
        from_var: &str,
        to_var: &str,
        n_points: usize,
    ) -> PyResult<(String, Vec<f64>, Vec<f64>, Vec<f64>)> {
        let c = self
            .inner
            .edge_curve(from_var, to_var, n_points)
            .map_err(PyValueError::new_err)?;
        Ok((c.edge_type, c.x_grid, c.y_centered, c.local_elasticity))
    }

    #[pyo3(signature = (cal_data, col_names, alpha=0.05))]
    fn calibrate_f32(
        &mut self,
        cal_data: PyReadonlyArray2<'_, f32>,
        col_names: Vec<String>,
        alpha: f64,
    ) -> PyResult<()> {
        let slice = cal_data
            .as_slice()
            .map_err(|_| PyValueError::new_err("Expected C-contiguous float32"))?;
        let view = MatrixView::new(slice, cal_data.shape()[0], cal_data.shape()[1], MatrixLayout::RowMajor)
            .map_err(PyValueError::new_err)?;
        self.inner
            .calibrate_conformal(view, &col_names, alpha)
            .map_err(PyValueError::new_err)
    }

    fn predict_f32(
        &self,
        data: PyReadonlyArray2<'_, f32>,
        col_names: Vec<String>,
        target_var: &str,
    ) -> PyResult<(Vec<f64>, Vec<f64>, f64)> {
        let slice = data
            .as_slice()
            .map_err(|_| PyValueError::new_err("Expected C-contiguous float32"))?;
        let view = MatrixView::new(slice, data.shape()[0], data.shape()[1], MatrixLayout::RowMajor)
            .map_err(PyValueError::new_err)?;
        let (actual_map, pred_map) = self
            .inner
            .predict_latents(view, &col_names)
            .map_err(PyValueError::new_err)?;
        let y_true = actual_map
            .get(target_var)
            .ok_or_else(|| PyValueError::new_err(format!("Target '{}' not found", target_var)))?
            .clone();
        let y_pred = pred_map
            .get(target_var)
            .ok_or_else(|| PyValueError::new_err(format!("Prediction '{}' not found", target_var)))?
            .clone();
        let hw = self
            .inner
            .calibrators
            .get(target_var)
            .map_or(0.0, |c| c.half_width());
        Ok((y_true, y_pred, hw))
    }

    fn intervene_f32(
        &self,
        data: PyReadonlyArray2<'_, f32>,
        col_names: Vec<String>,
        intervened_var: &str,
        delta: f64,
        target_var: &str,
    ) -> PyResult<(f64, f64, f64, f64)> {
        let slice = data
            .as_slice()
            .map_err(|_| PyValueError::new_err("Expected C-contiguous float32"))?;
        let view = MatrixView::new(slice, data.shape()[0], data.shape()[1], MatrixLayout::RowMajor)
            .map_err(PyValueError::new_err)?;
        let s = self
            .inner
            .intervene_shift(view, &col_names, intervened_var, delta, target_var)
            .map_err(PyValueError::new_err)?;
        Ok((s.total_effect, s.direct_effect, s.indirect_effect, s.conformal_half_width))
    }
}

#[pyfunction]
#[pyo3(signature = (formula, data, col_names, knots=6, smooth_lambda=1e-4, max_iter=300))]
fn fit_whitebox_f32(
    formula: &str,
    data: PyReadonlyArray2<'_, f32>,
    col_names: Vec<String>,
    knots: usize,
    smooth_lambda: f64,
    max_iter: usize,
) -> PyResult<PyWhiteBoxModel> {
    let slice = data
        .as_slice()
        .map_err(|_| PyValueError::new_err("Expected C-contiguous float32"))?;
    let view = MatrixView::new(slice, data.shape()[0], data.shape()[1], MatrixLayout::RowMajor)
        .map_err(PyValueError::new_err)?;
    let inner = fit_whitebox_sem(formula, view, &col_names, knots, smooth_lambda, max_iter)
        .map_err(PyValueError::new_err)?;
    Ok(PyWhiteBoxModel { inner })
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWhiteBoxModel>()?;
    m.add_function(wrap_pyfunction!(fit_whitebox_f32, m)?)?;
    Ok(())
}
