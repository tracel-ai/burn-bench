use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Practical hardware throughput limits measured on the current device.
///
/// These are the peaks a real kernel can realistically approach, used as the
/// denominator when reporting how much of the hardware a benchmark utilizes.
/// Each axis is optional so a provider that can only measure a subset (or a
/// future provider that fetches vendor specs) can leave the rest as `None`.
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct PracticalLimits {
    /// Peak memory bandwidth in bytes per second.
    pub mem_bytes_per_sec: Option<f64>,
    /// Peak non-tensor-core arithmetic throughput in FLOP per second.
    pub arith_flops_per_sec: Option<f64>,
    /// Peak tensor-core arithmetic throughput in FLOP per second.
    pub tensor_core_flops_per_sec: Option<f64>,
}

/// Fraction of each practical limit a benchmark reached.
///
/// Values are ratios (1.0 == 100%). A value can exceed 1.0: e.g. an arithmetic
/// utilization above 100% while tensor-core utilization stays below it is a
/// strong signal the kernel must be using the tensor cores.
#[derive(Debug, Default, Clone, Copy)]
pub struct Utilization {
    /// Achieved memory throughput as a fraction of the memory bandwidth peak.
    pub mem: Option<f64>,
    /// Achieved compute as a fraction of the arithmetic peak.
    pub arith: Option<f64>,
    /// Achieved compute as a fraction of the tensor-core peak.
    pub tensor_core: Option<f64>,
}

impl PracticalLimits {
    /// Compute the utilization of each practical limit for a benchmark that did
    /// `flops` floating-point operations and moved `bytes` bytes in a median
    /// time of `median`.
    ///
    /// Any axis is `None` when the corresponding benchmark property or measured
    /// peak is missing, or when the median duration is zero.
    pub fn utilization(
        &self,
        flops: Option<u64>,
        bytes: Option<u64>,
        median: Duration,
    ) -> Utilization {
        let secs = median.as_secs_f64();
        if secs <= 0.0 {
            return Utilization::default();
        }

        let ratio = |amount: Option<u64>, peak: Option<f64>| match (amount, peak) {
            (Some(amount), Some(peak)) if peak > 0.0 => Some((amount as f64 / secs) / peak),
            _ => None,
        };

        Utilization {
            mem: ratio(bytes, self.mem_bytes_per_sec),
            arith: ratio(flops, self.arith_flops_per_sec),
            tensor_core: ratio(flops, self.tensor_core_flops_per_sec),
        }
    }
}
