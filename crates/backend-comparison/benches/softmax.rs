use burn::tensor::{Distribution, Element, Shape, Tensor, activation::softmax, backend::Backend};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};
use derive_new::new;

#[derive(new)]
struct SoftmaxBenchmark<B: Backend, const D: usize> {
    shape: Shape,
    dim: usize,
    device: B::Device,
}

impl<B: Backend, const D: usize> Benchmark for SoftmaxBenchmark<B, D> {
    type Input = Tensor<B, D>;
    type Output = Tensor<B, D>;

    fn name(&self) -> String {
        format!("softmax-{:?}-{:?}", self.dim, B::FloatElem::dtype()).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.dims.clone()]
    }

    fn execute(&self, tensor: Self::Input) -> Self::Output {
        softmax(tensor, self.dim)
    }

    fn prepare(&self) -> Self::Input {
        Tensor::random(self.shape.clone(), Distribution::Default, &self.device)
    }

    fn sync(&self) {
        B::sync(&self.device).unwrap();
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) -> Vec<BenchmarkResult> {
    [
        (2, 6144, 6144),
        (4, 4096, 4096),
        (8, 2048, 2048),
        (16, 1024, 1024),
        (256, 256, 256),
    ]
    .into_iter()
    .map(|(a, b, c)| {
        let shape: Shape = [a, b, c].into();

        (0..shape.dims.len())
            .map(|dim| SoftmaxBenchmark::<B, 3>::new(shape.clone(), dim, device.clone()))
            .collect::<Vec<_>>()
    })
    .flatten()
    .map(|a| run_benchmark(a))
    .collect()
}

fn main() {
    burnbench::bench_on_backend!();
}
