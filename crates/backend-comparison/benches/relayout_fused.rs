use burn::tensor::{
    Device, Distribution, Shape, Tensor,
    module::{adaptive_avg_pool2d, avg_pool2d, interpolate, max_pool2d},
    ops::{InterpolateMode, InterpolateOptions},
};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};

pub struct RelayoutBenchmark {
    device: Device,
    shape: Shape,
    mode: RelayoutAlgorithm,
}

#[derive(Clone)]
pub enum RelayoutAlgorithm {
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

impl RelayoutAlgorithm {
    pub fn name(&self) -> String {
        match self {
            RelayoutAlgorithm::MaxPool2D(op) => format!(
                "max_pool2d_k{}_s{}_p{}_d{}",
                slice_to_string(&op.kernel_size),
                slice_to_string(&op.stride),
                slice_to_string(&op.padding),
                slice_to_string(&op.dilation)
            ),
            RelayoutAlgorithm::AvgPool2D(op) => format!(
                "avg_pool2d_k{}_s{}_p{}",
                slice_to_string(&op.kernel_size),
                slice_to_string(&op.stride),
                slice_to_string(&op.padding)
            ),
            RelayoutAlgorithm::AdaptivePool2d(op) => {
                format!("adaptive_avg_pool2d_o{}", slice_to_string(&op.output_size))
            }
            RelayoutAlgorithm::Interpolate(op) => format!("interpolate_{:?}", op.options.mode),
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

impl RelayoutBenchmark {
    pub fn execute(&self, input: Tensor<4>) -> Tensor<4> {
        self.mode.execute(input)
    }
}

impl RelayoutAlgorithm {
    pub fn execute(&self, input: Tensor<4>) -> Tensor<4> {
        match self {
            RelayoutAlgorithm::MaxPool2D(benchmark) => max_pool2d(
                input,
                benchmark.kernel_size,
                benchmark.stride,
                benchmark.padding,
                benchmark.dilation,
                false,
            ),
            RelayoutAlgorithm::AvgPool2D(benchmark) => avg_pool2d(
                input,
                benchmark.kernel_size,
                benchmark.stride,
                benchmark.padding,
                false,
                false,
            ),
            RelayoutAlgorithm::AdaptivePool2d(benchmark) => {
                adaptive_avg_pool2d(input, benchmark.output_size)
            }
            RelayoutAlgorithm::Interpolate(benchmark) => {
                interpolate(input, benchmark.output_size, benchmark.options.clone())
            }
        }
    }
}

impl Benchmark for RelayoutBenchmark {
    type Input = (Tensor<4>, Tensor<4>);
    type Output = Tensor<4>;

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
        let x = input.0;
        let zeros = input.1;

        // trigger relayout
        let x = x + zeros;

        // pool
        self.execute(x)
    }

    fn prepare(&self) -> Self::Input {
        let [batches, ch, h, w] = self.shape.dims();

        let x = Tensor::random([batches, h, w, ch], Distribution::Default, &self.device)
            .permute([0, 3, 1, 2]);
        let zeros = Tensor::zeros([batches, h, w, ch], &self.device).permute([0, 3, 1, 2]);

        (x, zeros)
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
        RelayoutAlgorithm::MaxPool2D(MaxPool2D {
            kernel_size: [5, 5],
            stride: [1, 1],
            padding: [2, 2],
            dilation: [1, 1],
        }),
        RelayoutAlgorithm::AvgPool2D(AvgPool2D {
            kernel_size: [5, 5],
            stride: [1, 1],
            padding: [2, 2],
        }),
        RelayoutAlgorithm::AdaptivePool2d(AdaptiveAvgPool2D {
            output_size: [256, 256],
        }),
        RelayoutAlgorithm::Interpolate(Interpolate {
            output_size: [1024, 1024],
            options: InterpolateOptions::new(InterpolateMode::Nearest),
        }),
        RelayoutAlgorithm::Interpolate(Interpolate {
            output_size: [256, 256],
            options: InterpolateOptions::new(InterpolateMode::Lanczos3),
        }),
    ];

    for mode in strategies {
        for shape in &shapes {
            benches.push(RelayoutBenchmark {
                shape: shape.clone(),
                device: device.clone(),
                mode: mode.clone(),
            });
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
