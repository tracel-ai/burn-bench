//! Shared dispatch helpers for the backend-comparison benchmarks.
//!
//! Since Burn dropped the `Backend` trait generic on tensors, backend selection
//! is no longer a compile-time type parameter chosen through feature flags.
//! Instead, every backend that can be compiled on the host is linked in (see the
//! OS-conditional `burn` dependency in `Cargo.toml`) and the concrete backend is
//! picked at runtime by injecting the right [`Device`].
//!
//! The runner passes the desired backend and dtype as runtime arguments
//! (`--device <label> --dtype <dtype>`); [`select_device`] / [`select_devices`]
//! turn those into a configured [`Device`].

use burn::tensor::{
    Device, DeviceConfig, DeviceError, DeviceIndex, DeviceKind, DeviceType, FloatDType,
};
use burnbench::__private::{get_argument, get_sharing_token, get_sharing_url, init_log};
use burnbench::{BenchmarkRecord, BenchmarkResult, BenchmarkSystemInfo, save_records};

/// Default backend label used when `--device` is not provided.
const DEFAULT_DEVICE: &str = "ndarray";

/// Returns the raw backend label passed via `--device` (e.g. `cuda`,
/// `tch-cpu`). The device selects the backend at runtime; it is independent
/// from the compile-time `--build` profile.
fn device_label() -> String {
    let args: Vec<String> = std::env::args().collect();
    get_argument(&args, "--device")
        .unwrap_or(DEFAULT_DEVICE)
        .to_string()
}

/// Returns the compile-time build profile requested via `--build` (e.g.
/// `default`, `no-fusion`). It is recorded in the result's `feature` column and
/// does not affect which device is selected.
fn build_label() -> String {
    let args: Vec<String> = std::env::args().collect();
    get_argument(&args, "--build")
        .unwrap_or("default")
        .to_string()
}

/// Returns the floating point dtype requested via `--dtype` (default `f32`).
fn dtype_arg() -> FloatDType {
    let args: Vec<String> = std::env::args().collect();
    match get_argument(&args, "--dtype") {
        Some("f16") => FloatDType::F16,
        Some("bf16") => FloatDType::BF16,
        Some("flex32") => FloatDType::Flex32,
        _ => FloatDType::F32,
    }
}

/// Applies the requested float dtype as the device default. This must happen
/// before any tensor is created on the device.
///
/// `AlreadyInitialized` is ignored so repeated calls are harmless, but an
/// unsupported dtype is surfaced as a panic: silently falling back to the
/// default would mislabel the results (the recorded dtype is read back from the
/// device settings).
fn configure_dtype(device: &mut Device, dtype: FloatDType) {
    match device.configure(DeviceConfig::default().float_dtype(dtype)) {
        Ok(()) | Err(DeviceError::AlreadyInitialized { .. }) => {}
        Err(err) => panic!("Failed to configure device dtype: {err}"),
    }
}

/// Builds the [`Device`] for the given base backend label.
///
/// Arms are gated by `cfg(target_os)` to match the backends that are actually
/// compiled in on each platform (see `Cargo.toml`). A label that is unknown or
/// unavailable on the host panics with a descriptive message.
fn build_device(base: &str) -> Device {
    match base {
        "wgpu" => Device::wgpu(DeviceKind::DefaultDevice),
        "webgpu" => Device::webgpu(DeviceKind::DefaultDevice),
        "cpu" => Device::cpu(),
        "flex" => Device::flex(),
        "ndarray"
        | "ndarray-simd"
        | "ndarray-blas-accelerate"
        | "ndarray-blas-netlib"
        | "ndarray-blas-openblas" => Device::ndarray(),
        #[cfg(feature = "tch")]
        "tch-cpu" => Device::libtorch(),
        #[cfg(feature = "tch")]
        "tch-cuda" => Device::libtorch_cuda(DeviceIndex::Default),
        #[cfg(feature = "tch")]
        "tch-metal" => Device::libtorch_mps(),
        #[cfg(not(target_os = "macos"))]
        "vulkan" => Device::vulkan(DeviceKind::DefaultDevice),
        #[cfg(not(target_os = "macos"))]
        "cuda" => Device::cuda(DeviceIndex::Default),
        #[cfg(target_os = "linux")]
        "rocm" => Device::rocm(DeviceIndex::Default),
        #[cfg(target_os = "macos")]
        "metal" => Device::metal(DeviceKind::DefaultDevice),
        other => panic!(
            "Backend `{other}` is not available on this platform. \
             Make sure it is enabled in the host's default features."
        ),
    }
}

/// Maps a device label to the [`DeviceType`] used for multi-device enumeration.
fn device_type(base: &str) -> DeviceType {
    // The `DeviceType` variants are gated by the same backend features as the
    // device factories, so the arms mirror `build_device`'s `cfg` gating.
    match base {
        "wgpu" => DeviceType::Wgpu,
        "webgpu" => DeviceType::WebGpu,
        "cpu" => DeviceType::Cpu,
        "flex" => DeviceType::Flex,
        base if base.starts_with("ndarray") => DeviceType::NdArray,
        #[cfg(feature = "tch")]
        base if base.starts_with("tch") => DeviceType::LibTorch,
        #[cfg(not(target_os = "macos"))]
        "vulkan" => DeviceType::Vulkan,
        #[cfg(not(target_os = "macos"))]
        "cuda" => DeviceType::Cuda,
        #[cfg(target_os = "linux")]
        "rocm" => DeviceType::Rocm,
        #[cfg(target_os = "macos")]
        "metal" => DeviceType::Metal,
        other => panic!("Backend `{other}` does not support multi-device benchmarks."),
    }
}

/// Selects and configures the [`Device`] requested by the runtime arguments.
pub fn select_device() -> Device {
    let _ = init_log();
    let mut device = build_device(&device_label());
    configure_dtype(&mut device, dtype_arg());
    device
}

/// Selects and configures every available [`Device`] of the requested backend,
/// for multi-device benchmarks.
pub fn select_devices() -> Vec<Device> {
    let _ = init_log();
    let dtype = dtype_arg();
    let mut devices: Vec<Device> = Device::enumerate(device_type(&device_label()))
        .iter()
        .cloned()
        .collect();
    for device in devices.iter_mut() {
        configure_dtype(device, dtype);
    }
    devices
}

/// Persists benchmark results, optionally sharing them with the server when a
/// sharing URL/token is provided via runtime arguments.
pub fn save(results: Vec<BenchmarkResult>, device: impl core::fmt::Debug) {
    let args: Vec<String> = std::env::args().collect();
    let url = get_sharing_url(&args);
    let token = get_sharing_token(&args);
    // The device selects the backend (runtime); the build profile is the
    // compile-time feature configuration (e.g. `no-fusion`).
    let backend = device_label();
    let feature = build_label();
    let burn_version =
        std::env::var("BURN_BENCH_BURN_VERSION").unwrap_or_else(|_| "main".to_string());
    let device_name = format!("{device:?}");

    let records: Vec<BenchmarkRecord> = results
        .into_iter()
        .map(|results| BenchmarkRecord {
            backend: backend.clone(),
            device: device_name.clone(),
            feature: feature.clone(),
            burn_version: burn_version.clone(),
            system_info: BenchmarkSystemInfo::new(),
            results,
        })
        .collect();

    save_records(records, url, token).unwrap();
}
