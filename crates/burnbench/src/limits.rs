use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Numeric dtype, mirrored from burn so that `burnbench` stays free of a `burn`
/// dependency (it also hosts the burn-free runner CLI). `backend-comparison` maps
/// `burn::tensor::DType` to and from this on the calibration seam.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    F64,
    F32,
    Flex32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
}

/// A typed piece of a benchmark's compute work. The dtype (and, for tensor-core
/// work, the hardware MMA tile) lets calibration measure the *real* peak for
/// exactly this configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Compute {
    /// Scalar/vector arithmetic operations performed in `dtype`.
    Arithmetic { dtype: DType, count: u64 },
    /// Tensor-core operations performed in `dtype`, issued as the given hardware
    /// MMA tile `[m, n, k]`.
    TensorCore {
        dtype: DType,
        mnk: [u32; 3],
        count: u64,
    },
}

impl Compute {
    fn dtype(&self) -> DType {
        match self {
            Compute::Arithmetic { dtype, .. } | Compute::TensorCore { dtype, .. } => *dtype,
        }
    }

    fn count(&self) -> u64 {
        match self {
            Compute::Arithmetic { count, .. } | Compute::TensorCore { count, .. } => *count,
        }
    }
}

/// A benchmark's declared best-case resource usage. Empty `compute` and `None`
/// `memory` means "not available" — the corresponding report columns show `N/A`.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Limit {
    /// Best-case bytes moved (reads + writes).
    pub memory: Option<u64>,
    /// Typed compute descriptors (a fused kernel may declare several).
    pub compute: Vec<Compute>,
}

/// Measured arithmetic peak (operations per second) for one dtype.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ArithPeak {
    pub dtype: DType,
    pub ops_per_sec: f64,
}

/// Measured tensor-core peak (operations per second) for one dtype + MMA tile.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TcPeak {
    pub dtype: DType,
    pub mnk: [u32; 3],
    pub ops_per_sec: f64,
}

/// Practical hardware peaks measured on a device, accumulated per configuration
/// across runs. Compute peaks are keyed by the exact configuration a benchmark
/// declared, so utilization is a direct measured ratio — no approximation.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Peaks {
    pub mem_bytes_per_sec: Option<f64>,
    pub arith: Vec<ArithPeak>,
    pub tensor_core: Vec<TcPeak>,
}

impl Peaks {
    /// Measured arithmetic peak (ops/s) for `dtype`, if known.
    pub fn arith(&self, dtype: DType) -> Option<f64> {
        self.arith
            .iter()
            .find(|p| p.dtype == dtype)
            .map(|p| p.ops_per_sec)
    }

    /// Measured tensor-core peak (ops/s) for `dtype` + `mnk`, if known.
    pub fn tensor_core(&self, dtype: DType, mnk: [u32; 3]) -> Option<f64> {
        self.tensor_core
            .iter()
            .find(|p| p.dtype == dtype && p.mnk == mnk)
            .map(|p| p.ops_per_sec)
    }

    /// Compute utilization of each practical limit for a benchmark whose declared
    /// usage is `limit` and whose median run time is `median`.
    ///
    /// - `mem`   = `(memory / t) / mem_peak`.
    /// - `arith` = Σ over ALL compute entries of `(count / t) / arith_peak(dtype)`
    ///   (tensor-core work also consumes arithmetic capacity, which is what makes
    ///   an arith% above 100% a signal that the tensor cores must be in use).
    /// - `tc`    = Σ over TensorCore entries of `(count / t) / tc_peak(dtype, mnk)`.
    ///
    /// An axis with no contributing measured peak is `None` (reported as `N/A`).
    pub fn utilization(&self, limit: &Limit, median: Duration) -> Utilization {
        let t = median.as_secs_f64();
        if t <= 0.0 {
            return Utilization::default();
        }

        let mem = match (limit.memory, self.mem_bytes_per_sec) {
            (Some(bytes), Some(peak)) if peak > 0.0 => Some((bytes as f64 / t) / peak),
            _ => None,
        };

        let mut arith = 0.0;
        let mut arith_any = false;
        let mut tc = 0.0;
        let mut tc_any = false;

        for entry in &limit.compute {
            let rate = entry.count() as f64 / t;
            if let Some(peak) = self.arith(entry.dtype())
                && peak > 0.0
            {
                arith += rate / peak;
                arith_any = true;
            }
            if let Compute::TensorCore { dtype, mnk, .. } = entry
                && let Some(peak) = self.tensor_core(*dtype, *mnk)
                && peak > 0.0
            {
                tc += rate / peak;
                tc_any = true;
            }
        }

        Utilization {
            mem,
            arith: arith_any.then_some(arith),
            tc: tc_any.then_some(tc),
        }
    }
}

/// Fraction of each practical limit a benchmark reached (1.0 == 100%). A value
/// can exceed 1.0: an arithmetic utilization above 100% while tensor-core
/// utilization stays near it is a strong signal the kernel must use the tensor
/// cores.
#[derive(Debug, Default, Clone, Copy)]
pub struct Utilization {
    pub mem: Option<f64>,
    pub arith: Option<f64>,
    pub tc: Option<f64>,
}

/// Source of the practical hardware peaks for a device.
///
/// The default implementation ([`crate`] does not provide one — it lives in
/// `backend-comparison`) measures each configuration with an on-device cubecl
/// kernel. A future implementation could fetch vendor spec sheets instead. Every
/// method returns `None` when a configuration cannot be measured (the backend has
/// no compute client, or the MMA tile/dtype is unsupported) — never a panic.
pub trait PeaksProvider {
    /// Stable identity of the device (independent of dtype), used as the cache key.
    fn device_key(&self) -> String;
    /// Peak memory bandwidth in bytes per second.
    fn measure_memory(&self) -> Option<f64>;
    /// Peak scalar/vector arithmetic in operations per second for `dtype`.
    fn measure_arith(&self, dtype: DType) -> Option<f64>;
    /// Peak tensor-core throughput in operations per second for `dtype` + MMA tile.
    fn measure_tensor_core(&self, dtype: DType, mnk: [u32; 3]) -> Option<f64>;
}

/// Assemble the peaks needed to score `limits`: load the device's cached [`Peaks`],
/// measure (via `provider`) only the configurations not already present, persist
/// the merged result, and return it.
///
/// Peaks accumulate on disk per device, so distinct configurations measured by
/// different benchmarks/runs are all reused. Unsupported configurations are simply
/// left absent (and cheaply re-checked next time).
pub fn resolve_peaks(provider: &impl PeaksProvider, limits: &[Limit]) -> Peaks {
    let key = provider.device_key();
    let mut peaks = load_cached(&key).unwrap_or_default();

    if peaks.mem_bytes_per_sec.is_none() && limits.iter().any(|l| l.memory.is_some()) {
        peaks.mem_bytes_per_sec = provider.measure_memory();
    }

    for limit in limits {
        for entry in &limit.compute {
            let dtype = entry.dtype();
            // Every compute entry is scored against the arithmetic peak.
            if peaks.arith(dtype).is_none()
                && let Some(ops) = provider.measure_arith(dtype)
            {
                peaks.arith.push(ArithPeak {
                    dtype,
                    ops_per_sec: ops,
                });
            }
            if let Compute::TensorCore { mnk, .. } = entry
                && peaks.tensor_core(dtype, *mnk).is_none()
                && let Some(ops) = provider.measure_tensor_core(dtype, *mnk)
            {
                peaks.tensor_core.push(TcPeak {
                    dtype,
                    mnk: *mnk,
                    ops_per_sec: ops,
                });
            }
        }
    }

    store_cached(&key, &peaks);
    peaks
}

fn cache_path(device_key: &str) -> Option<std::path::PathBuf> {
    let dir = dirs::home_dir()?
        .join(".cache")
        .join("burn")
        .join("burnbench");
    let sanitized: String = device_key
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    Some(dir.join(format!("peaks_{sanitized}.json")))
}

fn load_cached(device_key: &str) -> Option<Peaks> {
    let content = std::fs::read_to_string(cache_path(device_key)?).ok()?;
    serde_json::from_str(&content).ok()
}

fn store_cached(device_key: &str, peaks: &Peaks) {
    let Some(path) = cache_path(device_key) else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(peaks) {
        let _ = std::fs::write(path, json);
    }
}
