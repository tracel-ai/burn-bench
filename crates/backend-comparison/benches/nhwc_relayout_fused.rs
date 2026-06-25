use burn::tensor::{
    Device, Distribution, Shape, Tensor, Tolerance,
    module::{adaptive_avg_pool2d, avg_pool2d, interpolate, max_pool2d},
    ops::{InterpolateMode, InterpolateOptions},
};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};

#[derive(Clone)]
pub enum BenchmarkInput {
    Single4D(Tensor<4>, Tensor<4>),
}

#[derive(Clone)]
pub enum BenchmarkOutput {
    Dim4(Tensor<4>),
}

impl BenchmarkOutput {
    pub fn into_data(self) -> burn::tensor::TensorData {
        match self {
            BenchmarkOutput::Dim4(tensor) => tensor.into_data(),
        }
    }
}

pub struct NHWCRelayoutBenchmark {
    device: Device,
    shape: Shape,
    mode: NHWCRelayoutAlgorithm,
}

#[derive(Clone)]
pub enum NHWCRelayoutAlgorithm {
    MaxPool2D(MaxPool2D),
    AvgPool2D(AvgPool2D),
    AdaptivePool2d(AdaptiveAvgPool2D),
    Interpolate(Interpolate),
}

fn slice_to_string(slice: &[usize]) -> String {
    slice
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join("x")
}

impl NHWCRelayoutAlgorithm {
    pub fn name(&self) -> String {
        match self {
            NHWCRelayoutAlgorithm::MaxPool2D(op) => format!(
                "max_pool2d_k{}_s{}_p{}_d{}",
                slice_to_string(&op.kernel_size),
                slice_to_string(&op.stride),
                slice_to_string(&op.padding),
                slice_to_string(&op.dilation)
            ),
            NHWCRelayoutAlgorithm::AvgPool2D(op) => format!(
                "avg_pool2d_k{}_s{}_p{}",
                slice_to_string(&op.kernel_size),
                slice_to_string(&op.stride),
                slice_to_string(&op.padding)
            ),
            NHWCRelayoutAlgorithm::AdaptivePool2d(op) => {
                format!("adaptive_avg_pool2d_o{}", slice_to_string(&op.output_size))
            }
            NHWCRelayoutAlgorithm::Interpolate(op) => format!("interpolate_{:?}", op.options.mode),
        }
    }
}

#[derive(Clone, Copy)]
pub struct MaxPool2D {
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
}

#[derive(Clone, Copy)]
pub struct AvgPool2D {
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
}

#[derive(Clone, Copy)]
pub struct AdaptiveAvgPool2D {
    output_size: [usize; 2],
}

#[derive(Clone)]
pub struct Interpolate {
    output_size: [usize; 2],
    options: InterpolateOptions,
}

impl NHWCRelayoutBenchmark {
    pub fn bench(&self, input: BenchmarkInput) -> BenchmarkOutput {
        self.mode.execute(input, true)
    }

    #[cfg(feature = "correctness")]
    pub fn check_correctness(&self) {
        let input = self.prepare();
        let expected = self.mode.execute(input.clone(), false);
        let actual = self.bench(input.clone());
        expected
            .into_data()
            .assert_approx_eq(&actual.into_data(), Tolerance::<f32>::strict());
    }

    pub fn prepare_algorithm(&self) -> BenchmarkInput {
        self.mode.prepare(&self.shape, &self.device)
    }
}

impl NHWCRelayoutAlgorithm {
    pub fn execute(&self, input: BenchmarkInput, with_zeros: bool) -> BenchmarkOutput {
        match (self, input) {
            (NHWCRelayoutAlgorithm::MaxPool2D(benchmark), BenchmarkInput::Single4D(zeros, x)) => {
                let x = if with_zeros { x + zeros } else { x };
                BenchmarkOutput::Dim4(max_pool2d(
                    x,
                    benchmark.kernel_size,
                    benchmark.stride,
                    benchmark.padding,
                    benchmark.dilation,
                    false,
                ))
            }
            (NHWCRelayoutAlgorithm::AvgPool2D(benchmark), BenchmarkInput::Single4D(zeros, x)) => {
                let x = if with_zeros { x + zeros } else { x };
                BenchmarkOutput::Dim4(avg_pool2d(
                    x,
                    benchmark.kernel_size,
                    benchmark.stride,
                    benchmark.padding,
                    false,
                    false,
                ))
            }
            (
                NHWCRelayoutAlgorithm::AdaptivePool2d(benchmark),
                BenchmarkInput::Single4D(zeros, x),
            ) => {
                let x = if with_zeros { x + zeros } else { x };
                BenchmarkOutput::Dim4(adaptive_avg_pool2d(x, benchmark.output_size))
            }
            (NHWCRelayoutAlgorithm::Interpolate(benchmark), BenchmarkInput::Single4D(zeros, x)) => {
                let x = if with_zeros { x + zeros } else { x };
                BenchmarkOutput::Dim4(interpolate(
                    x,
                    benchmark.output_size,
                    benchmark.options.clone(),
                ))
            }
        }
    }

    fn prepare(&self, shape: &Shape, device: &Device) -> BenchmarkInput {
        match self {
            NHWCRelayoutAlgorithm::MaxPool2D(_)
            | NHWCRelayoutAlgorithm::AvgPool2D(_)
            | NHWCRelayoutAlgorithm::AdaptivePool2d(_)
            | NHWCRelayoutAlgorithm::Interpolate(_) => {
                let [batches, ch, h, w] = shape.as_slice() else {
                    panic!("shape must be 4D")
                };
                let x = Tensor::random([batches, h, w, ch], Distribution::Default, device)
                    .permute([0, 3, 1, 2]);
                let zeros = Tensor::zeros([batches, h, w, ch], device).permute([0, 3, 1, 2]);

                BenchmarkInput::Single4D(zeros, x)
            }
        }
    }
}

impl Benchmark for NHWCRelayoutBenchmark {
    type Input = BenchmarkInput;
    type Output = BenchmarkOutput;

    fn name(&self) -> String {
        format!(
            "{}_{:?}",
            self.mode.name(),
            self.device.settings().float_dtype
        )
        .to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }

    fn execute(&self, input: Self::Input) -> Self::Output {
        self.bench(input)
    }

    fn prepare(&self) -> Self::Input {
        self.prepare_algorithm()
    }

    fn sync(&self) {
        self.device.sync().unwrap();
    }
}

fn bench(device: &Device) -> Vec<BenchmarkResult> {
    let mut benches = Vec::new();
    let shapes: Vec<Shape> = vec![[2, 128, 512, 512], [2, 32, 512, 512]]
        .into_iter()
        .map(|s| s.into())
        .collect();

    let strategies = vec![
        NHWCRelayoutAlgorithm::MaxPool2D(MaxPool2D {
            kernel_size: [5, 5],
            stride: [1, 1],
            padding: [2, 2],
            dilation: [1, 1],
        }),
        NHWCRelayoutAlgorithm::AvgPool2D(AvgPool2D {
            kernel_size: [5, 5],
            stride: [1, 1],
            padding: [2, 2],
        }),
        NHWCRelayoutAlgorithm::AdaptivePool2d(AdaptiveAvgPool2D {
            output_size: [256, 256],
        }),
        NHWCRelayoutAlgorithm::Interpolate(Interpolate {
            output_size: [1024, 1024],
            options: InterpolateOptions::new(InterpolateMode::Nearest),
        }),
        NHWCRelayoutAlgorithm::Interpolate(Interpolate {
            output_size: [256, 256],
            options: InterpolateOptions::new(InterpolateMode::Lanczos3),
        }),
    ];

    for mode in strategies {
        for shape in &shapes {
            let bench = NHWCRelayoutBenchmark {
                shape: shape.clone(),
                device: device.clone(),
                mode: mode.clone(),
            };

            #[cfg(feature = "correctness")]
            bench.check_correctness();

            benches.push(bench);
        }
    }

    benches
        .into_iter()
        .map(|bench| run_benchmark(bench))
        .collect()
}

fn main() {
    let device = backend_comparison::select_device();
    let results = bench(&device);
    backend_comparison::save(results, &device);
}
