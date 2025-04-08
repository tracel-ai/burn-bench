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
            benches: Vec<burn_common::benchmark::BenchmarkResult>,
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

        $crate::bench_on_backend!(bench)
    }};

    ($fn_name:ident) => {{
        $crate::__private::init_log().unwrap();
        use std::env;

        #[cfg(feature = "cuda")]
        {
            use burn::backend::cuda::{Cuda, CudaDevice};
            use burn::tensor::f16;

            let device = CudaDevice::default();
            $crate::bench_on_backend!($fn_name, Cuda<f16>, device);
            $crate::bench_on_backend!($fn_name, Cuda<f32>, device);
        }

        // #[cfg(feature = "wgpu")]
        // {
        //     use burn::backend::wgpu::{Wgpu, WgpuDevice};

        //     $fn_name::<Wgpu<f32, i32>>(&WgpuDevice::default(), feature_name, url, token);
        // }

        // #[cfg(feature = "wgpu-spirv")]
        // {
        //     use burn::backend::wgpu::{Wgpu, WgpuDevice};
        //     use burn::tensor::f16;

        //     $fn_name::<Wgpu<f16, i32>>(&WgpuDevice::default(), feature_name, url, token);
        //     $fn_name::<Wgpu<f32, i32>>(&WgpuDevice::default(), feature_name, url, token);
        // }

        // #[cfg(feature = "metal")]
        // {
        //     use burn::backend::wgpu::{Wgpu, WgpuDevice};
        //     use burn::tensor::f16;

        //     $fn_name::<Wgpu<f16, i32>>(&WgpuDevice::default(), feature_name, url, token);
        //     $fn_name::<Wgpu<f32, i32>>(&WgpuDevice::default(), feature_name, url, token);
        // }

        // #[cfg(feature = "tch-gpu")]
        // {
        //     use burn::backend::{LibTorch, libtorch::LibTorchDevice};
        //     use burn::tensor::f16;

        //     #[cfg(not(target_os = "macos"))]
        //     let device = LibTorchDevice::Cuda(0);
        //     #[cfg(target_os = "macos")]
        //     let device = LibTorchDevice::Mps;

        //     let device: format!("{:?}", device),
        //     let result = $fn_name::<LibTorch<f16>>(&device);
        //     save_result(result, url, token, feature_name);
        // }

        // #[cfg(feature = "tch-cpu")]
        // {
        //     use burn::backend::{LibTorch, libtorch::LibTorchDevice};

        //     let device = LibTorchDevice::Cpu;
        //     $fn_name::<LibTorch>(&device, feature_name, url, token);
        // }

        // #[cfg(any(
        //     feature = "ndarray",
        //     feature = "ndarray-simd",
        //     feature = "ndarray-blas-netlib",
        //     feature = "ndarray-blas-openblas",
        //     feature = "ndarray-blas-accelerate",
        // ))]
        // {
        //     use burn::backend::NdArray;
        //     use burn::backend::ndarray::NdArrayDevice;

        //     let device = NdArrayDevice::Cpu;
        //     $fn_name::<NdArray>(&device, feature_name, url, token);
        // }

        // #[cfg(feature = "candle-cpu")]
        // {
        //     use burn::backend::Candle;
        //     use burn::backend::candle::CandleDevice;

        //     let device = CandleDevice::Cpu;
        //     $fn_name::<Candle>(&device, feature_name, url, token);
        // }

        // #[cfg(feature = "candle-cuda")]
        // {
        //     use burn::backend::Candle;
        //     use burn::backend::candle::CandleDevice;

        //     let device = CandleDevice::cuda(0);
        //     $fn_name::<Candle>(&device, feature_name, url, token);
        // }

        // #[cfg(feature = "candle-metal")]
        // {
        //     use burn::backend::Candle;
        //     use burn::backend::candle::CandleDevice;

        //     let device = CandleDevice::metal(0);
        //     $fn_name::<Candle>(&device, feature_name, url, token);
        // }

        // #[cfg(feature = "hip")]
        // {
        //     #[cfg(not(burn_version_lt_0170))]
        //     use burn::backend::hip::{Hip, HipDevice};
        //     #[cfg(burn_version_lt_0170)]
        //     use burn::backend::hip_jit::{Hip, HipDevice};
        //     use burn::tensor::f16;

        //     $fn_name::<Hip<f16>>(&HipDevice::default(), feature_name, url, token);
        //     $fn_name::<Hip<f32>>(&HipDevice::default(), feature_name, url, token);
        // }
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
        #[cfg(feature = "metal")]
        let feature_name = "metal";
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
        #[cfg(feature = "tch-gpu")]
        let feature_name = "tch-gpu";
        #[cfg(feature = "wgpu")]
        let feature_name = "wgpu";
        #[cfg(feature = "wgpu-fusion")]
        let feature_name = "wgpu-fusion";
        #[cfg(feature = "wgpu-spirv")]
        let feature_name = "wgpu-spirv";
        #[cfg(feature = "wgpu-spirv-fusion")]
        let feature_name = "wgpu-spirv-fusion";
        #[cfg(feature = "cuda")]
        let feature_name = "cuda";
        #[cfg(feature = "cuda-fusion")]
        let feature_name = "cuda-fusion";
        #[cfg(feature = "hip")]
        let feature_name = "hip";

        let device_name = format!("{:?}", &$device);
        let backend_name = <$backend as Backend>::name(&$device);
        let benches = $fn_name::<$backend>(&$device);
        __save_result(benches, backend_name, device_name, url, token, feature_name);
    };
}
