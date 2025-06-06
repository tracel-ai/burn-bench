use burn::tensor::{Distribution, Element, Shape, Tensor, backend::Backend, module::max_pool2d};
use burn_common::benchmark::{Benchmark, BenchmarkResult, run_benchmark};

pub struct MaxPool2dBenchmark<B: Backend> {
    shape: Shape,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    name: &'static str,
    device: B::Device,
}

impl<B: Backend> Benchmark for MaxPool2dBenchmark<B> {
    type Input = Tensor<B, 4>;
    type Output = Tensor<B, 4>;

    fn name(&self) -> String {
        format!("max_pool2d_{}-{:?}", self.name, B::FloatElem::dtype()).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.dims.clone()]
    }

    fn execute(&self, x: Self::Input) -> Self::Output {
        max_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )
    }

    fn prepare(&self) -> Self::Input {
        let [batches, ch, h, w] = self.shape.dims();
        Tensor::random([batches, h, w, ch], Distribution::Default, &self.device)
            .permute([0, 3, 1, 2])
    }

    fn sync(&self) {
        B::sync(&self.device)
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) -> Vec<BenchmarkResult> {
    let benchmark = MaxPool2dBenchmark::<B> {
        name: "default",
        shape: [32, 128, 512, 512].into(),
        kernel_size: [5, 5],
        stride: [2, 2],
        padding: [2, 2],
        dilation: [2, 2],
        device: device.clone(),
    };
    let benchmark2 = MaxPool2dBenchmark::<B> {
        name: "unit_stride",
        shape: [32, 32, 512, 512].into(),
        kernel_size: [5, 5],
        stride: [1, 1],
        padding: [2, 2],
        dilation: [1, 1],
        device: device.clone(),
    };

    vec![run_benchmark(benchmark), run_benchmark(benchmark2)]
}

fn main() {
    burnbench::bench_on_backend!();
}
