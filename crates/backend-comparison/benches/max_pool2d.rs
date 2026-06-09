use burn::tensor::{Device, Distribution, Shape, Tensor, module::max_pool2d};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};

pub struct MaxPool2dBenchmark {
    shape: Shape,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    name: &'static str,
    device: Device,
}

impl Benchmark for MaxPool2dBenchmark {
    type Input = Tensor<4>;
    type Output = Tensor<4>;

    fn name(&self) -> String {
        format!(
            "max_pool2d_{}-{:?}",
            self.name,
            self.device.settings().float_dtype
        )
        .to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }

    fn execute(&self, x: Self::Input) -> Self::Output {
        max_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            false,
        )
    }

    fn prepare(&self) -> Self::Input {
        let [batches, ch, h, w] = self.shape.dims();
        Tensor::random([batches, h, w, ch], Distribution::Default, &self.device)
            .permute([0, 3, 1, 2])
    }

    fn sync(&self) {
        self.device.sync().unwrap();
    }
}

#[allow(dead_code)]
fn bench(device: &Device) -> Vec<BenchmarkResult> {
    let benchmark = MaxPool2dBenchmark {
        name: "default",
        shape: [2, 128, 512, 512].into(),
        kernel_size: [5, 5],
        stride: [2, 2],
        padding: [2, 2],
        dilation: [2, 2],
        device: device.clone(),
    };
    let benchmark2 = MaxPool2dBenchmark {
        name: "unit_stride",
        shape: [2, 32, 512, 512].into(),
        kernel_size: [5, 5],
        stride: [1, 1],
        padding: [2, 2],
        dilation: [1, 1],
        device: device.clone(),
    };

    vec![run_benchmark(benchmark), run_benchmark(benchmark2)]
}

fn main() {
    let device = backend_comparison::select_device();
    let results = bench(&device);
    backend_comparison::save(results, &device);
}
