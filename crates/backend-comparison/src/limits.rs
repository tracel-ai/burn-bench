//! On-device measurement of practical hardware peaks, implemented with cubecl
//! kernels. A `cmma` micro-kernel measures the tensor-core peak for an exact MMA
//! tile + dtype; an FMA-loop kernel measures the scalar arithmetic peak; a
//! high-level elementwise pass measures memory bandwidth. Results are assembled
//! (and cached) by [`burnbench::resolve_peaks`].
//!
//! Every measurement returns `None` when it cannot be performed — the backend has
//! no cubecl compute client (CPU / ndarray / tch), or the requested MMA tile/dtype
//! is not supported by the hardware — so the report shows `N/A` rather than
//! crashing.

use core::mem::size_of;
use std::time::Instant;

use burn::backend::DispatchDevice;
use burn::tensor::{Device, Distribution, Tensor};
use burnbench::{DType, PeaksProvider};
use cubecl::features::MmaConfig;
use cubecl::future::block_on;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use half::{bf16, f16};

/// Measures practical peaks on `device` using cubecl kernels.
pub struct CalibrationProvider {
    pub device: Device,
}

impl PeaksProvider for CalibrationProvider {
    fn device_key(&self) -> String {
        format!("{:?}", self.device)
    }

    fn measure_memory(&self) -> Option<f64> {
        // Bandwidth is dtype-agnostic; measured with a cheap high-level elementwise
        // pass (1 read + 1 write per element) on whatever backend the device uses.
        let n = 4096usize;
        let elem = burn::tensor::DType::from(self.device.settings().float_dtype).size() as f64;
        let bytes = 2.0 * (n * n) as f64 * elem;

        let x: Tensor<2> = Tensor::random([n, n], Distribution::Default, &self.device);
        for _ in 0..5 {
            let _ = x.clone().mul_scalar(1.0001);
        }
        self.device.sync().ok()?;

        let reps = 20;
        let start = Instant::now();
        for _ in 0..reps {
            let _ = x.clone().mul_scalar(1.0001);
        }
        self.device.sync().ok()?;
        let secs = start.elapsed().as_secs_f64() / reps as f64;

        (secs > 0.0).then(|| bytes / secs)
    }

    fn measure_arith(&self, dtype: DType) -> Option<f64> {
        match self.device.as_dispatch() {
            #[cfg(not(target_os = "macos"))]
            DispatchDevice::Cuda(d) => arith_on(cubecl::cuda::CudaRuntime::client(d), dtype),
            #[cfg(not(target_os = "macos"))]
            DispatchDevice::Vulkan(d) => arith_on(
                cubecl::wgpu::WgpuRuntime::<cubecl::wgpu::AutoCompiler>::client(d),
                dtype,
            ),
            DispatchDevice::Wgpu(d) => arith_on(
                cubecl::wgpu::WgpuRuntime::<cubecl::wgpu::AutoCompiler>::client(d),
                dtype,
            ),
            DispatchDevice::WebGpu(d) => arith_on(
                cubecl::wgpu::WgpuRuntime::<cubecl::wgpu::AutoCompiler>::client(d),
                dtype,
            ),
            #[cfg(target_os = "macos")]
            DispatchDevice::Metal(d) => arith_on(
                cubecl::wgpu::WgpuRuntime::<cubecl::wgpu::AutoCompiler>::client(d),
                dtype,
            ),
            _ => None,
        }
    }

    fn measure_tensor_core(&self, dtype: DType, mnk: [u32; 3]) -> Option<f64> {
        // v1 supports only the canonical 16x16x16 tile with F16/BF16 inputs and an
        // F32 accumulator; any other configuration reports N/A.
        if mnk != [16, 16, 16] {
            return None;
        }
        let a_type = match dtype {
            DType::F16 => ElemType::Float(FloatKind::F16),
            DType::BF16 => ElemType::Float(FloatKind::BF16),
            _ => return None,
        };
        let cfg = MmaConfig {
            a_type: a_type.into(),
            b_type: a_type.into(),
            cd_type: ElemType::Float(FloatKind::F32).into(),
            m: 16,
            n: 16,
            k: 16,
        };

        match self.device.as_dispatch() {
            #[cfg(not(target_os = "macos"))]
            DispatchDevice::Cuda(d) => tc_on(cubecl::cuda::CudaRuntime::client(d), dtype, &cfg),
            #[cfg(not(target_os = "macos"))]
            DispatchDevice::Vulkan(d) => tc_on(
                cubecl::wgpu::WgpuRuntime::<cubecl::wgpu::AutoCompiler>::client(d),
                dtype,
                &cfg,
            ),
            DispatchDevice::Wgpu(d) => tc_on(
                cubecl::wgpu::WgpuRuntime::<cubecl::wgpu::AutoCompiler>::client(d),
                dtype,
                &cfg,
            ),
            DispatchDevice::WebGpu(d) => tc_on(
                cubecl::wgpu::WgpuRuntime::<cubecl::wgpu::AutoCompiler>::client(d),
                dtype,
                &cfg,
            ),
            #[cfg(target_os = "macos")]
            DispatchDevice::Metal(d) => tc_on(
                cubecl::wgpu::WgpuRuntime::<cubecl::wgpu::AutoCompiler>::client(d),
                dtype,
                &cfg,
            ),
            _ => None,
        }
    }
}

// --- Runtime-generic dispatch --------------------------------------------------

fn arith_on<R: Runtime>(client: ComputeClient<R>, dtype: DType) -> Option<f64> {
    Some(match dtype {
        DType::F16 => arith_run::<R, f16>(&client),
        DType::BF16 => arith_run::<R, bf16>(&client),
        DType::F32 => arith_run::<R, f32>(&client),
        DType::Flex32 => arith_run::<R, f32>(&client),
        // Integer arithmetic peaks are out of scope for v1.
        _ => return None,
    })
}

fn tc_on<R: Runtime>(client: ComputeClient<R>, dtype: DType, cfg: &MmaConfig) -> Option<f64> {
    if !client.features().matmul.cmma.contains(cfg) {
        return None;
    }
    Some(match dtype {
        DType::F16 => tc_run::<R, f16>(&client),
        DType::BF16 => tc_run::<R, bf16>(&client),
        _ => return None,
    })
}

// --- Measurement --------------------------------------------------------------

const ARITH_CUBE_DIM: u32 = 256;
const ARITH_CUBES: u32 = 512;
const ARITH_ITERS: u32 = 32;
const ARITH_ACC: f64 = 8.0; // independent accumulator chains in the kernel

const TC_CUBES: u32 = 4096;
const TC_ITERS: u32 = 16;

fn arith_run<R: Runtime, F: Float + CubeElement>(client: &ComputeClient<R>) -> f64 {
    let threads = (ARITH_CUBES * ARITH_CUBE_DIM) as usize;
    let out = client.empty(threads * size_of::<F>());

    let launch = || unsafe {
        arith_peak_kernel::launch::<F, R>(
            client,
            CubeCount::Static(ARITH_CUBES, 1, 1),
            CubeDim::new_1d(ARITH_CUBE_DIM),
            BufferArg::from_raw_parts(out.clone(), threads),
            ARITH_ITERS,
        )
    };

    let secs = time_launches(client, 5, 100, launch);
    // 2 FLOPs per FMA, ARITH_ACC chains, ARITH_ITERS iterations, per thread.
    let ops = 2.0 * ARITH_ACC * ARITH_ITERS as f64 * threads as f64;
    ops / secs
}

fn tc_run<R: Runtime, F: Float + CubeElement>(client: &ComputeClient<R>) -> f64 {
    let lhs = client.empty(256 * size_of::<F>());
    let rhs = client.empty(256 * size_of::<F>());
    let out = client.empty(256 * size_of::<f32>());

    let launch = || unsafe {
        tc_peak_kernel::launch::<F, R>(
            client,
            CubeCount::Static(TC_CUBES, 1, 1),
            CubeDim::new_1d(32),
            BufferArg::from_raw_parts(lhs.clone(), 256),
            BufferArg::from_raw_parts(rhs.clone(), 256),
            BufferArg::from_raw_parts(out.clone(), 256),
            TC_ITERS,
        )
    };

    let secs = time_launches(client, 5, 50, launch);
    // 2 * m * n * k FLOPs per tile, one tile per plane per iteration.
    let ops = 2.0 * 16.0 * 16.0 * 16.0 * TC_CUBES as f64 * TC_ITERS as f64;
    ops / secs
}

/// Runs `launch` `warmup` times, then times `reps` launches, returning the mean
/// seconds per launch.
fn time_launches<R: Runtime>(
    client: &ComputeClient<R>,
    warmup: u32,
    reps: u32,
    mut launch: impl FnMut(),
) -> f64 {
    for _ in 0..warmup {
        launch();
    }
    let _ = block_on(client.sync());

    let start = Instant::now();
    for _ in 0..reps {
        launch();
    }
    let _ = block_on(client.sync());
    start.elapsed().as_secs_f64() / reps as f64
}

// --- Kernels ------------------------------------------------------------------

#[cube(launch)]
fn arith_peak_kernel<F: Float>(output: &mut [F], #[comptime] iters: u32) {
    let b = F::new(1.000_000_1);
    let c = F::new(0.000_000_1);

    // Eight independent accumulator chains to expose instruction-level
    // parallelism (a single dependent chain would be latency-bound).
    let mut a0 = F::new(1.0);
    let mut a1 = F::new(1.1);
    let mut a2 = F::new(1.2);
    let mut a3 = F::new(1.3);
    let mut a4 = F::new(1.4);
    let mut a5 = F::new(1.5);
    let mut a6 = F::new(1.6);
    let mut a7 = F::new(1.7);

    #[unroll]
    for _ in 0..iters {
        a0 = a0 * b + c;
        a1 = a1 * b + c;
        a2 = a2 * b + c;
        a3 = a3 * b + c;
        a4 = a4 * b + c;
        a5 = a5 * b + c;
        a6 = a6 * b + c;
        a7 = a7 * b + c;
    }

    output[ABSOLUTE_POS] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

#[cube(launch)]
fn tc_peak_kernel<F: Float>(lhs: &[F], rhs: &[F], out: &mut [f32], #[comptime] iters: u32) {
    let a = cmma::Matrix::<F>::from_slice(
        cmma::MatrixIdent::A,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::RowMajor,
        lhs,
        16,
    );
    let b = cmma::Matrix::<F>::from_slice(
        cmma::MatrixIdent::B,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::ColMajor,
        rhs,
        16,
    );
    let c = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );

    // Loop-carried on the accumulator so the executes can't be folded away.
    #[unroll]
    for _ in 0..iters {
        cmma::execute(&a, &b, &c, &c);
    }

    cmma::store(out, &c, 16, cmma::MatrixLayout::RowMajor);
}
