use burn::tensor::{
    Device, Distribution, Shape, Tensor,
    module::{adaptive_avg_pool2d, avg_pool2d, max_pool2d},
};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};

pub struct PoolBenchmark {
    name: String,
    device: Device,
    shape: Shape,
    mode: PoolMode,
}

#[derive(Clone, Copy)]
pub enum PoolMode {
    MaxPool2D(MaxPool2D),
    AvgPool2D(AvgPool2D),
    AdaptivePool2d(AdaptiveAvgPool2D),
}

impl PoolMode {
    pub fn name(&self) -> &'static str {
        match self {
            PoolMode::MaxPool2D(_) => "max_pool2d",
            PoolMode::AvgPool2D(_) => "avg_pool2d",
            PoolMode::AdaptivePool2d(_) => "adaptive_avg_pool2d",
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

impl PoolBenchmark {
    pub fn execute(&self, input: Tensor<4>) -> Tensor<4> {
        self.mode.execute(input)
    }
}

impl PoolMode {
    pub fn execute(&self, input: Tensor<4>) -> Tensor<4> {
        match self {
            PoolMode::MaxPool2D(benchmark) => max_pool2d(
                input,
                benchmark.kernel_size,
                benchmark.stride,
                benchmark.padding,
                benchmark.dilation,
                false,
            ),
            PoolMode::AvgPool2D(benchmark) => avg_pool2d(
                input,
                benchmark.kernel_size,
                benchmark.stride,
                benchmark.padding,
                false,
                false,
            ),
            PoolMode::AdaptivePool2d(benchmark) => {
                adaptive_avg_pool2d(input, benchmark.output_size)
            }
        }
    }
}

impl Benchmark for PoolBenchmark {
    type Input = (Tensor<4>, Tensor<4>);
    type Output = Tensor<4>;

    fn name(&self) -> String {
        format!(
            "{}_{}-{:?}",
            self.mode.name(),
            self.name,
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
        PoolMode::MaxPool2D(MaxPool2D {
            kernel_size: [5, 5],
            stride: [2, 2],
            padding: [2, 2],
            dilation: [2, 2],
        }),
        PoolMode::MaxPool2D(MaxPool2D {
            kernel_size: [5, 5],
            stride: [1, 1],
            padding: [2, 2],
            dilation: [1, 1],
        }),
        PoolMode::AvgPool2D(AvgPool2D {
            kernel_size: [5, 5],
            stride: [2, 2],
            padding: [2, 2],
        }),
        PoolMode::AvgPool2D(AvgPool2D {
            kernel_size: [5, 5],
            stride: [1, 1],
            padding: [2, 2],
        }),
        PoolMode::AdaptivePool2d(AdaptiveAvgPool2D {
            output_size: [256, 256],
        }),
        PoolMode::AdaptivePool2d(AdaptiveAvgPool2D {
            output_size: [256, 256],
        }),
    ];

    for (name, mode) in strategies
        .iter()
        .enumerate()
        .map(|(i, m)| (format!("strategy_{}", i), m.clone()))
    {
        for shape in &shapes {
            benches.push(PoolBenchmark {
                name: name.clone(),
                shape: shape.clone(),
                device: device.clone(),
                mode: mode,
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
