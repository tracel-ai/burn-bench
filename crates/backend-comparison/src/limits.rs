//! On-device measurement of practical hardware peaks, using only high-level burn
//! tensor operations (no direct cubecl dependency — burn is tracked at
//! `branch = "main"`, so pinning a separate cubecl rev would risk two divergent
//! cubecl copies).
//!
//! - **Tensor-core peak**: a large square matmul in the device's dtype. burn's
//!   autotuned matmul lowers to the hardware's cooperative-matrix (tensor-core)
//!   kernels for the dtypes that support them, so its throughput is a real
//!   practical tensor-core peak. The declared MMA tile (`mnk`) is descriptive
//!   metadata here — autotune selects the actual tile.
//! - **Arithmetic peak**: a compute-heavy fused elementwise chain (many FMAs per
//!   element, few memory accesses).
//! - **Memory bandwidth**: a cheap elementwise pass (1 read + 1 write / element).
//!
//! Peaks are measured in the device's *configured* dtype, so a request for a
//! different dtype (or an integer dtype) returns `None` (reported as `N/A`).
//! Results are assembled and cached by [`burnbench::resolve_peaks`].

use std::time::Instant;

use burn::tensor::{DType as BurnDType, Device, Distribution, FloatDType, Tensor};
use burnbench::{DType, PeaksProvider};

/// Number of timed samples per calibration measurement (results are cached).
const SAMPLES: usize = 20;
const WARMUP: usize = 5;

/// Measures practical peaks on `device` with high-level burn tensor operations.
pub struct CalibrationProvider {
    pub device: Device,
}

impl CalibrationProvider {
    /// The device's configured float dtype, in burn-free form.
    fn float_dtype(&self) -> DType {
        to_dtype(self.device.settings().float_dtype)
    }

    /// Byte size of one element of the device's configured float dtype.
    fn elem_size(&self) -> f64 {
        BurnDType::from(self.device.settings().float_dtype).size() as f64
    }
}

impl PeaksProvider for CalibrationProvider {
    fn device_key(&self) -> String {
        format!("{:?}", self.device)
    }

    fn measure_memory(&self) -> Option<f64> {
        let n = 4096usize;
        let bytes = 2.0 * (n * n) as f64 * self.elem_size();
        let x: Tensor<2> = Tensor::random([n, n], Distribution::Default, &self.device);
        let secs = time(&self.device, || {
            let _ = x.clone().mul_scalar(1.0001);
        });
        (secs > 0.0).then_some(bytes / secs)
    }

    fn measure_arith(&self, dtype: DType) -> Option<f64> {
        // High-level ops run in the device's configured dtype only.
        if dtype != self.float_dtype() {
            return None;
        }
        let n = 4096usize;
        let iters = 64usize;
        let x: Tensor<2> = Tensor::random([n, n], Distribution::Default, &self.device);
        let secs = time(&self.device, || {
            // A fused multiply-add chain: with kernel fusion this collapses into a
            // single compute-bound kernel (2 FLOPs per iteration per element).
            let mut y = x.clone();
            for _ in 0..iters {
                y = y.mul_scalar(1.000_000_1).add_scalar(0.000_000_1);
            }
            let _ = y;
        });
        let ops = 2.0 * iters as f64 * (n * n) as f64;
        (secs > 0.0).then_some(ops / secs)
    }

    fn measure_tensor_core(&self, dtype: DType, _mnk: [u32; 3]) -> Option<f64> {
        if dtype != self.float_dtype() {
            return None;
        }
        let n = 2048usize;
        let lhs: Tensor<2> = Tensor::random([n, n], Distribution::Default, &self.device);
        let rhs: Tensor<2> = Tensor::random([n, n], Distribution::Default, &self.device);
        let secs = time(&self.device, || {
            let _ = lhs.clone().matmul(rhs.clone());
        });
        let ops = 2.0 * (n as f64).powi(3);
        (secs > 0.0).then_some(ops / secs)
    }
}

/// Runs `run` `WARMUP` times, then times `SAMPLES` runs (syncing the device
/// around the timed region), returning the mean seconds per run.
fn time(device: &Device, mut run: impl FnMut()) -> f64 {
    for _ in 0..WARMUP {
        run();
    }
    let _ = device.sync();

    let start = Instant::now();
    for _ in 0..SAMPLES {
        run();
    }
    let _ = device.sync();
    start.elapsed().as_secs_f64() / SAMPLES as f64
}

/// Maps a burn float dtype to the burn-free dtype used in peak/limit records.
fn to_dtype(dtype: FloatDType) -> DType {
    match dtype {
        FloatDType::F64 => DType::F64,
        FloatDType::F32 => DType::F32,
        FloatDType::Flex32 => DType::Flex32,
        FloatDType::F16 => DType::F16,
        FloatDType::BF16 => DType::BF16,
    }
}
