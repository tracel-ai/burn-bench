use tracing_subscriber::{
    Layer,
    filter::{LevelFilter, filter_fn},
    layer::SubscriberExt,
    registry,
    util::SubscriberInitExt,
};

/// Simple parse to retrieve additional argument passed to cargo bench command
/// We cannot use clap here as clap parser does not allow to have unknown arguments.
pub fn get_argument<'a>(args: &'a [String], arg_name: &'a str) -> Option<&'a str> {
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            arg if arg == arg_name && i + 1 < args.len() => {
                return Some(&args[i + 1]);
            }
            _ => i += 1,
        }
    }
    None
}

/// Specialized function to retrieve the sharing token
pub fn get_sharing_token(args: &[String]) -> Option<&str> {
    get_argument(args, "--sharing-token")
}

/// Specialized function to retrieve the sharing URL
pub fn get_sharing_url(args: &[String]) -> Option<&str> {
    get_argument(args, "--sharing-url")
}

pub fn init_log() -> Result<(), String> {
    let layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_filter(LevelFilter::INFO)
        .with_filter(filter_fn(|m| {
            if let Some(path) = m.module_path() {
                if path.starts_with("wgpu") {
                    return false;
                }
            }
            true
        }));

    let result = registry().with(layer).try_init();

    if result.is_err() {
        update_panic_hook();
    }

    result.map_err(|err| format!("{err:?}"))
}

fn update_panic_hook() {
    let hook = std::panic::take_hook();

    std::panic::set_hook(Box::new(move |info| {
        log::error!("PANIC => {}", info);
        hook(info);
    }));
}

#[macro_export]
macro_rules! define_types {
    () => {
        pub fn __save_result(
            benches: Vec<$crate::BenchmarkResult>,
            backend_name: String,
            device: String,
            url: Option<&str>,
            token: Option<&str>,
            feature: &str,
        ) {
            let burn_version =
                std::env::var("BURN_BENCH_BURN_VERSION").unwrap_or_else(|_| "main".to_string());

            let records: Vec<$crate::BenchmarkRecord> = benches
                .into_iter()
                .map(|bench| $crate::BenchmarkRecord {
                    backend: backend_name.clone(),
                    device: device.clone(),
                    feature: feature.to_string(),
                    burn_version: burn_version.clone(),
                    system_info: $crate::BenchmarkSystemInfo::new(),
                    results: $crate::BenchmarkResult {
                        raw: $crate::BenchmarkDurations {
                            timing_method: Default::default(),
                            durations: bench.raw.durations,
                        },
                        computed: $crate::BenchmarkComputations {
                            mean: bench.computed.mean,
                            median: bench.computed.median,
                            variance: bench.computed.variance,
                            min: bench.computed.min,
                            max: bench.computed.max,
                        },
                        git_hash: bench.git_hash,
                        name: bench.name,
                        options: bench.options,
                        shapes: bench.shapes,
                        timestamp: bench.timestamp,
                    },
                })
                .collect();

            $crate::save_records(records, url, token).unwrap()
        }
    };
}

#[macro_export]
macro_rules! bench_on_backend {
    () => {{
        $crate::define_types!();

        #[cfg(feature = "f32")]
        {
            $crate::bench_on_backend!(bench, f32)
        }

        #[cfg(feature = "f16")]
        {
            $crate::bench_on_backend!(bench, burn::tensor::f16)
        }

        #[cfg(feature = "bf16")]
        {
            $crate::bench_on_backend!(bench, burn::tensor::bf16)
        }

        #[cfg(feature = "flex32")]
        {
            $crate::bench_on_backend!(bench, burn::tensor::flex32)
        }
    }};

    ($fn_name:ident, $dtype:ty) => {{
        $crate::__private::init_log().unwrap();
        use std::env;

        #[cfg(feature = "cuda")]
        {
            #[cfg(not(feature = "legacy-v16"))]
            use burn::backend::Cuda;
            #[cfg(feature = "legacy-v16")]
            use burn::backend::CudaJit as Cuda;

            let device = Default::default();
            $crate::bench_on_backend!($fn_name, Cuda<$dtype>, device);
        }

        #[cfg(feature = "metal")]
        {
            use burn::backend::Metal;

            let device = Default::default();
            $crate::bench_on_backend!($fn_name, Metal<$dtype>, device);
        }

        #[cfg(all(target_os = "macos", feature = "vulkan"))]
        {
            panic!("vulkan benchmarks are not supported on macOS, use the wgpu backend instead.");
        }

        #[cfg(any(feature = "wgpu", feature = "vulkan"))]
        {
            use burn::backend::Wgpu;

            let device = Default::default();
            $crate::bench_on_backend!($fn_name, Wgpu<$dtype>, device);
        }

        #[cfg(feature = "ndarray")]
        {
            use burn::backend::NdArray;

            let device = Default::default();
            $crate::bench_on_backend!($fn_name, NdArray<$dtype>, device);
        }

        #[cfg(feature = "tch-cpu")]
        {
            use burn::backend::{LibTorch, libtorch::LibTorchDevice};

            let device = LibTorchDevice::Cpu;
            $crate::bench_on_backend!($fn_name, LibTorch<$dtype>, device);
        }

        #[cfg(feature = "tch-cuda")]
        {
            use burn::backend::{LibTorch, libtorch::LibTorchDevice};

            let device = LibTorchDevice::Cuda(0);
            $crate::bench_on_backend!($fn_name, LibTorch<$dtype>, device);
        }

        #[cfg(feature = "tch-metal")]
        {
            use burn::backend::{LibTorch, libtorch::LibTorchDevice};

            let device = LibTorchDevice::Mps;
            $crate::bench_on_backend!($fn_name, LibTorch<$dtype>, device);
        }

        #[cfg(feature = "candle-cpu")]
        {
            use burn::backend::candle::{Candle, CandleDevice};

            let device = CandleDevice::Cpu;
            $crate::bench_on_backend!($fn_name, Candle<$dtype>, device);
        }

        #[cfg(feature = "candle-cuda")]
        {
            use burn::backend::candle::{Candle, CandleDevice};

            let device = CandleDevice::cuda(0);
            $crate::bench_on_backend!($fn_name, Candle<$dtype>, device);
        }

        #[cfg(feature = "candle-metal")]
        {
            use burn::backend::candle::{Candle, CandleDevice};

            let device = CandleDevice::metal(0);
            $crate::bench_on_backend!($fn_name, Candle<$dtype>, device);
        }

        #[cfg(feature = "rocm")]
        {
            #[cfg(feature = "legacy-v16")]
            use burn::backend::HipJit as Hip;
            #[cfg(not(feature = "legacy-v16"))]
            use burn::backend::Rocm;

            let device = Default::default();
            $crate::bench_on_backend!($fn_name, Rocm<$dtype>, device);
        }
    }};

    ($fn_name:ident, $backend:ty, $device:ident) => {
        let args: Vec<String> = env::args().collect();
        let url = $crate::__private::get_sharing_url(&args);
        let token = $crate::__private::get_sharing_token(&args);

        #[cfg(feature = "candle-accelerate")]
        let feature_name = "candle-accelerate";
        #[cfg(feature = "candle-cpu")]
        let feature_name = "candle-cpu";
        #[cfg(feature = "candle-cuda")]
        let feature_name = "candle-cuda";
        #[cfg(feature = "candle-metal")]
        let feature_name = "candle-metal";
        #[cfg(feature = "ndarray")]
        let feature_name = "ndarray";
        #[cfg(feature = "ndarray-simd")]
        let feature_name = "ndarray-simd";
        #[cfg(feature = "ndarray-blas-accelerate")]
        let feature_name = "ndarray-blas-accelerate";
        #[cfg(feature = "ndarray-blas-netlib")]
        let feature_name = "ndarray-blas-netlib";
        #[cfg(feature = "ndarray-blas-openblas")]
        let feature_name = "ndarray-blas-openblas";
        #[cfg(feature = "tch-cpu")]
        let feature_name = "tch-cpu";
        #[cfg(feature = "tch-cuda")]
        let feature_name = "tch-cuda";
        #[cfg(feature = "tch-metal")]
        let feature_name = "tch-metal";
        #[cfg(feature = "wgpu")]
        let feature_name = "wgpu";
        #[cfg(feature = "wgpu-fusion")]
        let feature_name = "wgpu-fusion";
        #[cfg(feature = "vulkan")]
        let feature_name = "vulkan";
        #[cfg(feature = "vulkan-fusion")]
        let feature_name = "vulkan-fusion";
        #[cfg(feature = "metal")]
        let feature_name = "metal";
        #[cfg(feature = "metal-fusion")]
        let feature_name = "metal-fusion";
        #[cfg(feature = "cuda")]
        let feature_name = "cuda";
        #[cfg(feature = "cuda-fusion")]
        let feature_name = "cuda-fusion";
        #[cfg(feature = "rocm")]
        let feature_name = "rocm";
        #[cfg(feature = "rocm-fusion")]
        let feature_name = "rocm-fusion";

        let device_name = format!("{:?}", &$device);
        #[cfg(not(feature = "legacy-v16"))]
        let backend_name = <$backend as Backend>::name(&$device);
        #[cfg(feature = "legacy-v16")]
        let backend_name = <$backend as Backend>::name();
        let benches = $fn_name::<$backend>(&$device);
        __save_result(benches, backend_name, device_name, url, token, feature_name);
    };
}
