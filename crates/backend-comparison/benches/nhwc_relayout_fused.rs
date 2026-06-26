#[cfg(feature = "correctness")]
use burn::tensor::Tolerance;
use burn::tensor::{
    Device, Distribution, Tensor, TensorData,
    module::{
        adaptive_avg_pool1d, adaptive_avg_pool2d, avg_pool1d, avg_pool2d, interpolate, max_pool1d,
        max_pool1d_with_indices, max_pool2d, max_pool2d_with_indices,
    },
    ops::{InterpolateMode, InterpolateOptions},
};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};

impl Benchmark for NHWCRelayoutBenchmark {
    type Input = BenchmarkInput;
    type Output = BenchmarkOutput;

    fn name(&self) -> String {
        format!(
            "{}_{:?}",
            self.op.name(),
            self.device.settings().float_dtype
        )
        .to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        self.op.shapes()
    }

    fn execute(&self, input: Self::Input) -> Self::Output {
        self.op.run(input, &self.device, true)
    }

    fn prepare(&self) -> Self::Input {
        self.op.prepare(&self.device)
    }

    fn prepare_cloned(&self) -> bool {
        false
    }

    fn sync(&self) {
        self.device.sync().unwrap();
    }
}

fn bench(device: &Device) -> Vec<BenchmarkResult> {
    let forward_ops: Vec<Box<dyn RelayoutOp>> = vec![
        Box::new(AvgPool1d {
            shape: [2, 4096, 4096],
            kernel_size: 4,
            stride: 4,
            padding: 0,
        }),
        Box::new(AvgPool2d {
            shape: [2, 64, 512, 512],
            kernel_size: [3, 3],
            stride: [1, 1],
            padding: [1, 1],
        }),
        Box::new(AdaptiveAvgPool1d {
            shape: [2, 512, 24576],
            output_size: 1024,
        }),
        Box::new(AdaptiveAvgPool2d {
            shape: [2, 64, 512, 512],
            output_size: [128, 128],
        }),
        Box::new(MaxPool1d {
            shape: [2, 5120, 4096],
            kernel_size: 4,
            stride: 4,
            padding: 0,
            dilation: 1,
            with_indices: false,
        }),
        Box::new(MaxPool1d {
            shape: [2, 4096, 4096],
            kernel_size: 4,
            stride: 4,
            padding: 0,
            dilation: 1,
            with_indices: true,
        }),
        Box::new(MaxPool2d {
            shape: [2, 64, 448, 448],
            kernel_size: [3, 3],
            stride: [1, 1],
            padding: [1, 1],
            dilation: [1, 1],
            with_indices: false,
        }),
        Box::new(MaxPool2d {
            shape: [2, 64, 448, 448],
            kernel_size: [3, 3],
            stride: [1, 1],
            padding: [1, 1],
            dilation: [1, 1],
            with_indices: true,
        }),
        Box::new(Interpolate {
            shape: [2, 256, 128, 128],
            output_size: [384, 384],
            mode: InterpolateMode::Nearest,
        }),
        Box::new(Interpolate {
            shape: [2, 256, 128, 128],
            output_size: [384, 384],
            mode: InterpolateMode::Bilinear,
        }),
    ];

    let mut forward_results = Vec::new();
    run_ops(forward_ops, device, &mut forward_results);

    forward_results
}

fn main() {
    let device = backend_comparison::select_device();

    let forward_results = bench(&device);

    backend_comparison::save(forward_results, &device);
}

/// Cloneable transport for the prepared input tensors of a relayout op.
#[derive(Clone)]
pub enum BenchmarkInput {
    F3(Vec<Tensor<3>>),
    F4(Vec<Tensor<4>>),
}

#[derive(Clone)]
pub enum BenchmarkOutput {
    D1(Tensor<1>),
    D3(Tensor<3>),
    D4(Tensor<4>),
}

impl BenchmarkOutput {
    pub fn into_data(self) -> TensorData {
        match self {
            BenchmarkOutput::D1(t) => t.into_data(),
            BenchmarkOutput::D3(t) => t.into_data(),
            BenchmarkOutput::D4(t) => t.into_data(),
        }
    }
}

/// A single relayout operation to benchmark.
trait RelayoutOp {
    fn name(&self) -> String;
    fn shapes(&self) -> Vec<Vec<usize>>;
    fn prepare(&self, device: &Device) -> BenchmarkInput;
    fn run(&self, input: BenchmarkInput, device: &Device, with_zeros: bool) -> BenchmarkOutput;
}

/// Optionally prepend an elementwise `+ zeros` op. This is what the relayout
/// optimization fuses the NCHW<->NHWC permutation into.
fn fuse_relayout<const D: usize>(t: Tensor<D>, with_zeros: bool, device: &Device) -> Tensor<D> {
    if with_zeros {
        let zeros = Tensor::zeros(t.shape(), device);
        t + zeros
    } else {
        t
    }
}

/// Random `[N, C, L]` tensor laid out in memory as NLC (the 1D analogue of NHWC).
fn rand_nlc(shape: [usize; 3], device: &Device) -> Tensor<3> {
    let [n, c, l] = shape;
    Tensor::random([n, l, c], Distribution::Default, device).permute([0, 2, 1])
}

/// Random `[N, C, H, W]` tensor laid out in memory as NHWC.
fn rand_nhwc(shape: [usize; 4], device: &Device) -> Tensor<4> {
    let [n, c, h, w] = shape;
    Tensor::random([n, h, w, c], Distribution::Default, device).permute([0, 3, 1, 2])
}

fn slice_to_string(slice: &[usize]) -> String {
    slice
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join("x")
}

struct AvgPool1d {
    shape: [usize; 3],
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl RelayoutOp for AvgPool1d {
    fn name(&self) -> String {
        format!(
            "avg_pool1d_k{}_s{}_p{}",
            self.kernel_size, self.stride, self.padding
        )
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }
    fn prepare(&self, device: &Device) -> BenchmarkInput {
        BenchmarkInput::F3(vec![rand_nlc(self.shape, device)])
    }
    fn run(&self, input: BenchmarkInput, device: &Device, with_zeros: bool) -> BenchmarkOutput {
        let BenchmarkInput::F3(mut v) = input else {
            unreachable!()
        };
        let x = fuse_relayout(v.pop().unwrap(), with_zeros, device);
        BenchmarkOutput::D3(avg_pool1d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            false,
            false,
        ))
    }
}

struct AvgPool2d {
    shape: [usize; 4],
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
}

impl RelayoutOp for AvgPool2d {
    fn name(&self) -> String {
        format!(
            "avg_pool2d_k{}_s{}_p{}",
            slice_to_string(&self.kernel_size),
            slice_to_string(&self.stride),
            slice_to_string(&self.padding)
        )
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }
    fn prepare(&self, device: &Device) -> BenchmarkInput {
        BenchmarkInput::F4(vec![rand_nhwc(self.shape, device)])
    }
    fn run(&self, input: BenchmarkInput, device: &Device, with_zeros: bool) -> BenchmarkOutput {
        let BenchmarkInput::F4(mut v) = input else {
            unreachable!()
        };
        let x = fuse_relayout(v.pop().unwrap(), with_zeros, device);
        BenchmarkOutput::D4(avg_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            false,
            false,
        ))
    }
}

struct AdaptiveAvgPool1d {
    shape: [usize; 3],
    output_size: usize,
}

impl RelayoutOp for AdaptiveAvgPool1d {
    fn name(&self) -> String {
        format!("adaptive_avg_pool1d_o{}", self.output_size)
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }
    fn prepare(&self, device: &Device) -> BenchmarkInput {
        BenchmarkInput::F3(vec![rand_nlc(self.shape, device)])
    }
    fn run(&self, input: BenchmarkInput, device: &Device, with_zeros: bool) -> BenchmarkOutput {
        let BenchmarkInput::F3(mut v) = input else {
            unreachable!()
        };
        let x = fuse_relayout(v.pop().unwrap(), with_zeros, device);
        BenchmarkOutput::D3(adaptive_avg_pool1d(x, self.output_size))
    }
}

struct AdaptiveAvgPool2d {
    shape: [usize; 4],
    output_size: [usize; 2],
}

impl RelayoutOp for AdaptiveAvgPool2d {
    fn name(&self) -> String {
        format!(
            "adaptive_avg_pool2d_o{}",
            slice_to_string(&self.output_size)
        )
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }
    fn prepare(&self, device: &Device) -> BenchmarkInput {
        BenchmarkInput::F4(vec![rand_nhwc(self.shape, device)])
    }
    fn run(&self, input: BenchmarkInput, device: &Device, with_zeros: bool) -> BenchmarkOutput {
        let BenchmarkInput::F4(mut v) = input else {
            unreachable!()
        };
        let x = fuse_relayout(v.pop().unwrap(), with_zeros, device);
        BenchmarkOutput::D4(adaptive_avg_pool2d(x, self.output_size))
    }
}

struct MaxPool1d {
    shape: [usize; 3],
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    with_indices: bool,
}

impl RelayoutOp for MaxPool1d {
    fn name(&self) -> String {
        let suffix = if self.with_indices {
            "_with_indices"
        } else {
            ""
        };
        format!(
            "max_pool1d{}_k{}_s{}_p{}_d{}",
            suffix, self.kernel_size, self.stride, self.padding, self.dilation
        )
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }
    fn prepare(&self, device: &Device) -> BenchmarkInput {
        BenchmarkInput::F3(vec![rand_nlc(self.shape, device)])
    }
    fn run(&self, input: BenchmarkInput, device: &Device, with_zeros: bool) -> BenchmarkOutput {
        let BenchmarkInput::F3(mut v) = input else {
            unreachable!()
        };
        let x = fuse_relayout(v.pop().unwrap(), with_zeros, device);
        let out = if self.with_indices {
            max_pool1d_with_indices(
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                false,
            )
            .0
        } else {
            max_pool1d(
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                false,
            )
        };
        BenchmarkOutput::D3(out)
    }
}

struct MaxPool2d {
    shape: [usize; 4],
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    with_indices: bool,
}

impl RelayoutOp for MaxPool2d {
    fn name(&self) -> String {
        let suffix = if self.with_indices {
            "_with_indices"
        } else {
            ""
        };
        format!(
            "max_pool2d{}_k{}_s{}_p{}_d{}",
            suffix,
            slice_to_string(&self.kernel_size),
            slice_to_string(&self.stride),
            slice_to_string(&self.padding),
            slice_to_string(&self.dilation)
        )
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }
    fn prepare(&self, device: &Device) -> BenchmarkInput {
        BenchmarkInput::F4(vec![rand_nhwc(self.shape, device)])
    }
    fn run(&self, input: BenchmarkInput, device: &Device, with_zeros: bool) -> BenchmarkOutput {
        let BenchmarkInput::F4(mut v) = input else {
            unreachable!()
        };
        let x = fuse_relayout(v.pop().unwrap(), with_zeros, device);
        let out = if self.with_indices {
            max_pool2d_with_indices(
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                false,
            )
            .0
        } else {
            max_pool2d(
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                false,
            )
        };
        BenchmarkOutput::D4(out)
    }
}

struct Interpolate {
    shape: [usize; 4],
    output_size: [usize; 2],
    mode: InterpolateMode,
}

impl RelayoutOp for Interpolate {
    fn name(&self) -> String {
        format!(
            "interpolate_{:?}_o{}",
            self.mode,
            slice_to_string(&self.output_size)
        )
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }
    fn prepare(&self, device: &Device) -> BenchmarkInput {
        BenchmarkInput::F4(vec![rand_nhwc(self.shape, device)])
    }
    fn run(&self, input: BenchmarkInput, device: &Device, with_zeros: bool) -> BenchmarkOutput {
        let BenchmarkInput::F4(mut v) = input else {
            unreachable!()
        };
        let x = fuse_relayout(v.pop().unwrap(), with_zeros, device);
        BenchmarkOutput::D4(interpolate(
            x,
            self.output_size,
            InterpolateOptions::new(self.mode.clone()),
        ))
    }
}

pub struct NHWCRelayoutBenchmark {
    device: Device,
    op: Box<dyn RelayoutOp>,
}

const SEED: u64 = 420;

impl NHWCRelayoutBenchmark {
    /// Run the op with the relayout fusion (`+ zeros`) and without it, and assert
    /// the outputs match. A mismatch means the fused relayout path is wrong.
    #[cfg(feature = "correctness")]
    fn check_correctness(&self) {
        self.device.seed(SEED);
        let input_fused = self.op.prepare(&self.device);
        let fused = self.op.run(input_fused, &self.device, true);

        self.device.seed(SEED);
        let input_ref = self.op.prepare(&self.device);
        let reference = self.op.run(input_ref, &self.device, false);

        fused
            .into_data()
            .assert_approx_eq(&reference.into_data(), Tolerance::<f32>::balanced());
    }
}

fn run_ops(ops: Vec<Box<dyn RelayoutOp>>, device: &Device, results: &mut Vec<BenchmarkResult>) {
    for op in ops {
        let bench = NHWCRelayoutBenchmark {
            device: device.clone(),
            op,
        };

        #[cfg(feature = "correctness")]
        bench.check_correctness();

        results.push(run_benchmark(bench));
    }
}
