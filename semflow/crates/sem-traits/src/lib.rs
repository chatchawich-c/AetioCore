/// Trait สำหรับเลขทศนิยมที่รองรับ Zero-Copy ทั้งฝั่ง Python (f32/f64) และ R (f64)
pub trait SemFloat: Copy + Send + Sync + 'static {
    fn to_f64(self) -> f64;
    fn from_f64(v: f64) -> Self;
}

impl SemFloat for f32 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

impl SemFloat for f64 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v
    }
}

/// รูปแบบการจัดเรียงหน่วยความจำของเมทริกซ์ต้นทาง
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixLayout {
    RowMajor, // Python / NumPy C-Contiguous
    ColMajor, // R matrix / Fortran-Contiguous
}

/// มุมมองเมทริกซ์แบบ Zero-Copy 100% ที่ใช้ร่วมกันได้ทุกภาษาและทุก Crate
#[derive(Debug, Clone, Copy)]
pub struct MatrixView<'a, T: SemFloat> {
    data: &'a [T],
    nrows: usize,
    ncols: usize,
    layout: MatrixLayout,
}

impl<'a, T: SemFloat> MatrixView<'a, T> {
    pub fn new(
        data: &'a [T],
        nrows: usize,
        ncols: usize,
        layout: MatrixLayout,
    ) -> Result<Self, String> {
        if data.len() != nrows * ncols {
            return Err(format!(
                "Buffer length {} does not match {}x{}",
                data.len(),
                nrows,
                ncols
            ));
        }
        Ok(Self {
            data,
            nrows,
            ncols,
            layout,
        })
    }

    #[inline(always)]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline(always)]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        let idx = match self.layout {
            MatrixLayout::RowMajor => row * self.ncols + col,
            MatrixLayout::ColMajor => col * self.nrows + row,
        };
        self.data[idx].to_f64()
    }
}

// =========================================================================
// Plugin Contracts สำหรับ semflow-dl และ Domain Plugins (รูปแบบที่ 2)
// =========================================================================

/// ตัวดำเนินการบนเส้นเชื่อมหรือโหนดในกราฟ DAG (เช่น Linear, I-Spline, B-Spline)
pub trait StructuralOperator: Send + Sync {
    fn name(&self) -> &str;
    fn eval_1d(&self, x: f64, params: &[f64]) -> f64;
    fn deriv_1d(&self, x: f64, params: &[f64]) -> f64;
}

/// ฟังก์ชัน Loss เสริม (เช่น L1/L2 RegSEM, Arrhenius Physics Residual)
pub trait ObjectiveLoss: Send + Sync {
    fn name(&self) -> &str;
    fn penalty(&self, params: &[f64]) -> f64;
    fn grad_penalty(&self, params: &[f64], grad_out: &mut [f64]);
}

/// ตัวปรับเทียบความไม่แน่นอนปลายทาง (เช่น Split-Conformal, Online ACI)
pub trait Calibrator: Send + Sync {
    fn calibrate(&mut self, residuals: &[f64], alpha: f64);
    fn half_width(&self) -> f64;
}
