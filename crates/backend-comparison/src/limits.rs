//! On-device measurement of the practical throughput limits used to report how
//! much of the hardware each benchmark utilizes.
//!
//! The limits are obtained behind the [`LimitsProvider`] seam. The default
//! [`CalibrationProvider`] runs a few tiny micro-benchmarks on the target
//! device; a future provider could instead fetch vendor peak specs. Results are
//! cached on disk per (device, dtype) so the many bench-binary invocations that
//! make up a run don't each re-measure.

use std::path::PathBuf;

use burn::tensor::{Device, Distribution, FloatDType, Tensor};
use burnbench::{Benchmark, BenchmarkComputations, PracticalLimits, TimingMethod};

/// Number of measured samples per calibration micro-benchmark. Kept small since
/// the result is cached and reused across the whole run.
const CALIB_SAMPLES: usize = 8;

/// Size in bytes of one element of the given float dtype.
pub fn dtype_size(dtype: FloatDType) -> usize {
    match dtype {
        FloatDType::F16 | FloatDType::BF16 => 2,
        _ => 4,
    }
}

/// Source of the practical hardware limits for a device.
pub trait LimitsProvider {
    fn measure(&self, device: &Device) -> PracticalLimits;
}

/// Default provider: measures each limit with a simple on-device micro-benchmark.
pub struct CalibrationProvider;

impl LimitsProvider for CalibrationProvider {
    fn measure(&self, device: &Device) -> PracticalLimits {
        let dtype = device.settings().float_dtype;
        let elem = dtype_size(dtype) as f64;

        // Peak memory bandwidth: a cheap elementwise pass moves 1 read + 1 write
        // per element, so it is bound by bandwidth rather than compute.
        let mem = {
            let n = 4096;
            let median = median_secs(MemBench {
                n,
                device: device.clone(),
            });
            let bytes = 2.0 * (n * n) as f64 * elem;
            per_sec(bytes, median)
        };

        // Peak tensor-core throughput: a large square matmul in the device dtype
        // (engages the tensor cores when the hardware/dtype supports it).
        let tensor_core = {
            let n = 2048;
            let median = median_secs(MatmulBench {
                n,
                device: device.clone(),
            });
            let flops = 2.0 * (n as f64).powi(3);
            per_sec(flops, median)
        };

        // Peak non-tensor-core arithmetic: a fused elementwise chain doing many
        // FMA-like ops per element. This is an approximation (its accuracy
        // depends on fusion collapsing the chain into one compute-bound kernel);
        // it is the seam a future spec-fetching provider would replace.
        let arith = {
            let n = 4096;
            let iters = 32;
            let median = median_secs(ArithBench {
                n,
                iters,
                device: device.clone(),
            });
            let flops = 2.0 * iters as f64 * (n * n) as f64;
            per_sec(flops, median)
        };

        PracticalLimits {
            mem_bytes_per_sec: mem,
            arith_flops_per_sec: arith,
            tensor_core_flops_per_sec: tensor_core,
        }
    }
}

/// Returns the practical limits for `device`, using a disk cache keyed by the
/// device and its configured dtype. Set `BURN_BENCH_LIMITS=off` to skip
/// calibration entirely (all axes reported as `N/A`), or
/// `BURN_BENCH_RECALIBRATE=1` to force a fresh measurement.
pub fn measure_limits(device: &Device) -> PracticalLimits {
    if matches!(std::env::var("BURN_BENCH_LIMITS").as_deref(), Ok("off")) {
        return PracticalLimits::default();
    }

    let recalibrate = std::env::var("BURN_BENCH_RECALIBRATE").is_ok();
    if !recalibrate && let Some(cached) = load_cached(device) {
        return cached;
    }

    let limits = CalibrationProvider.measure(device);
    store_cached(device, &limits);
    limits
}

fn per_sec(amount: f64, secs: f64) -> Option<f64> {
    if secs > 0.0 {
        Some(amount / secs)
    } else {
        None
    }
}

fn median_secs<B: Benchmark>(bench: B) -> f64 {
    let durations = bench.run(TimingMethod::System);
    BenchmarkComputations::new(&durations).median.as_secs_f64()
}

fn cache_path(device: &Device) -> Option<PathBuf> {
    let dir = dirs::home_dir()?
        .join(".cache")
        .join("burn")
        .join("burnbench");
    let raw = format!("{device:?}-{:?}", device.settings().float_dtype);
    let key: String = raw
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    Some(dir.join(format!("limits_{key}.json")))
}

fn load_cached(device: &Device) -> Option<PracticalLimits> {
    let path = cache_path(device)?;
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

fn store_cached(device: &Device, limits: &PracticalLimits) {
    let Some(path) = cache_path(device) else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(limits) {
        let _ = std::fs::write(path, json);
    }
}

// --- Calibration micro-benchmarks ---------------------------------------------

struct MemBench {
    n: usize,
    device: Device,
}

impl Benchmark for MemBench {
    type Input = Tensor<2>;
    type Output = Tensor<2>;

    fn name(&self) -> String {
        "calib-mem".to_string()
    }

    fn num_samples(&self) -> usize {
        CALIB_SAMPLES
    }

    fn prepare(&self) -> Self::Input {
        Tensor::random([self.n, self.n], Distribution::Default, &self.device)
    }

    fn execute(&self, input: Self::Input) -> Self::Output {
        input.mul_scalar(1.0001)
    }

    fn sync(&self) {
        self.device.sync().unwrap();
    }
}

struct MatmulBench {
    n: usize,
    device: Device,
}

impl Benchmark for MatmulBench {
    type Input = (Tensor<2>, Tensor<2>);
    type Output = Tensor<2>;

    fn name(&self) -> String {
        "calib-matmul".to_string()
    }

    fn num_samples(&self) -> usize {
        CALIB_SAMPLES
    }

    fn prepare(&self) -> Self::Input {
        let lhs = Tensor::random([self.n, self.n], Distribution::Default, &self.device);
        let rhs = Tensor::random([self.n, self.n], Distribution::Default, &self.device);
        (lhs, rhs)
    }

    fn execute(&self, (lhs, rhs): Self::Input) -> Self::Output {
        lhs.matmul(rhs)
    }

    fn sync(&self) {
        self.device.sync().unwrap();
    }
}

struct ArithBench {
    n: usize,
    iters: usize,
    device: Device,
}

impl Benchmark for ArithBench {
    type Input = Tensor<2>;
    type Output = Tensor<2>;

    fn name(&self) -> String {
        "calib-arith".to_string()
    }

    fn num_samples(&self) -> usize {
        CALIB_SAMPLES
    }

    fn prepare(&self) -> Self::Input {
        Tensor::random([self.n, self.n], Distribution::Default, &self.device)
    }

    fn execute(&self, input: Self::Input) -> Self::Output {
        // Each iteration is one multiply + one add (2 FLOPs per element). The
        // scalars are near-identity to keep the values bounded without changing
        // the FLOP count.
        let mut x = input;
        for _ in 0..self.iters {
            x = x.mul_scalar(1.0000001).add_scalar(0.0000001);
        }
        x
    }

    fn sync(&self) {
        self.device.sync().unwrap();
    }
}
