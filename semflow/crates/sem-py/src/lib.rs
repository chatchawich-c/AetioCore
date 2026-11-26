use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use sem_traits::{MatrixLayout, MatrixView};

#[pyclass]
#[derive(Clone)]
struct PyPlsResult {
    #[pyo3(get)]
    consistent: bool,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    converged: bool,
    #[pyo3(get)]
    paths: Vec<(String, String, f64, Option<f64>, Option<f64>)>,
    #[pyo3(get)]
    loadings: Vec<(String, String, f64)>,
    #[pyo3(get)]
    metrics: Vec<(String, String, f64, f64, f64, Option<f64>)>,
    #[pyo3(get)]
    htmt: Vec<(String, String, f64)>,
}

#[pyclass]
#[derive(Clone)]
struct PyCbResult {
    #[pyo3(get)]
    converged: bool,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    f_ml: f64,
    #[pyo3(get)]
    params: Vec<(String, String, String, f64, f64, bool)>,
    #[pyo3(get)]
    chi_square: f64,
    #[pyo3(get)]
    df: usize,
    #[pyo3(get)]
    cfi: f64,
    #[pyo3(get)]
    tli: f64,
    #[pyo3(get)]
    rmsea: f64,
    #[pyo3(get)]
    srmr: f64,
}

fn wrap_pls(r: sem_pls::PlsResult) -> PyPlsResult {
    PyPlsResult {
        consistent: r.consistent,
        iterations: r.iterations,
        converged: r.converged,
        paths: r
            .paths
            .into_iter()
            .map(|p| (p.from, p.to, p.estimate, p.se, p.t_stat))
            .collect(),
        loadings: r.loadings,
        metrics: r
            .metrics
            .into_iter()
            .map(|m| (m.latent, m.mode, m.rho_a, m.ave, m.cr, m.r_squared))
            .collect(),
        htmt: r.htmt,
    }
}

fn wrap_cb(r: sem_cb::CbResult) -> PyCbResult {
    PyCbResult {
        converged: r.converged,
        iterations: r.iterations,
        f_ml: r.f_ml,
        params: r
            .params
            .into_iter()
            .map(|p| (p.lhs, p.op, p.rhs, p.estimate, p.se, p.fixed))
            .collect(),
        chi_square: r.fit.chi_square,
        df: r.fit.df,
        cfi: r.fit.cfi,
        tli: r.fit.tli,
        rmsea: r.fit.rmsea,
        srmr: r.fit.srmr,
    }
}

#[pyfunction]
#[pyo3(signature = (formula, data, col_names, consistent=false, n_boot=0, max_iter=300, tol=1e-6))]
fn fit_pls_f32(
    formula: &str,
    data: PyReadonlyArray2<'_, f32>,
    col_names: Vec<String>,
    consistent: bool,
    n_boot: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<PyPlsResult> {
    let ir = sem_dsl::parse_formula(formula).map_err(PyValueError::new_err)?;
    let slice = data
        .as_slice()
        .map_err(|_| PyValueError::new_err("Array must be C-contiguous float32"))?;
    let view = MatrixView::new(slice, data.shape()[0], data.shape()[1], MatrixLayout::RowMajor)
        .map_err(PyValueError::new_err)?;
    let res = sem_pls::fit_pls(&ir, view, &col_names, consistent, n_boot, max_iter, tol)
        .map_err(PyValueError::new_err)?;
    Ok(wrap_pls(res))
}

#[pyfunction]
#[pyo3(signature = (formula, data, col_names, consistent=false, n_boot=0, max_iter=300, tol=1e-6))]
fn fit_pls_f64(
    formula: &str,
    data: PyReadonlyArray2<'_, f64>,
    col_names: Vec<String>,
    consistent: bool,
    n_boot: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<PyPlsResult> {
    let ir = sem_dsl::parse_formula(formula).map_err(PyValueError::new_err)?;
    let slice = data
        .as_slice()
        .map_err(|_| PyValueError::new_err("Array must be C-contiguous float64"))?;
    let view = MatrixView::new(slice, data.shape()[0], data.shape()[1], MatrixLayout::RowMajor)
        .map_err(PyValueError::new_err)?;
    let res = sem_pls::fit_pls(&ir, view, &col_names, consistent, n_boot, max_iter, tol)
        .map_err(PyValueError::new_err)?;
    Ok(wrap_pls(res))
}

#[pyfunction]
#[pyo3(signature = (formula, data, col_names, max_iter=500))]
fn fit_cb_f32(
    formula: &str,
    data: PyReadonlyArray2<'_, f32>,
    col_names: Vec<String>,
    max_iter: usize,
) -> PyResult<PyCbResult> {
    let ir = sem_dsl::parse_formula(formula).map_err(PyValueError::new_err)?;
    let slice = data
        .as_slice()
        .map_err(|_| PyValueError::new_err("Array must be C-contiguous float32"))?;
    let view = MatrixView::new(slice, data.shape()[0], data.shape()[1], MatrixLayout::RowMajor)
        .map_err(PyValueError::new_err)?;
    let res = sem_cb::fit_cb_ml(&ir, view, &col_names, max_iter).map_err(PyValueError::new_err)?;
    Ok(wrap_cb(res))
}

#[pyfunction]
#[pyo3(signature = (formula, data, col_names, max_iter=500))]
fn fit_cb_f64(
    formula: &str,
    data: PyReadonlyArray2<'_, f64>,
    col_names: Vec<String>,
    max_iter: usize,
) -> PyResult<PyCbResult> {
    let ir = sem_dsl::parse_formula(formula).map_err(PyValueError::new_err)?;
    let slice = data
        .as_slice()
        .map_err(|_| PyValueError::new_err("Array must be C-contiguous float64"))?;
    let view = MatrixView::new(slice, data.shape()[0], data.shape()[1], MatrixLayout::RowMajor)
        .map_err(PyValueError::new_err)?;
    let res = sem_cb::fit_cb_ml(&ir, view, &col_names, max_iter).map_err(PyValueError::new_err)?;
    Ok(wrap_cb(res))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPlsResult>()?;
    m.add_class::<PyCbResult>()?;
    m.add_function(wrap_pyfunction!(fit_pls_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fit_pls_f64, m)?)?;
    m.add_function(wrap_pyfunction!(fit_cb_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fit_cb_f64, m)?)?;
    Ok(())
}
