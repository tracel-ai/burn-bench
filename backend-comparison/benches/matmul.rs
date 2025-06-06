use burn::tensor::{Distribution, Element, Shape, Tensor, backend::Backend};
use burn_common::benchmark::{Benchmark, BenchmarkResult, run_benchmark};
use derive_new::new;

#[derive(new)]
struct MatmulBenchmark<B: Backend, const D: usize> {
    shape_lhs: Shape,
    shape_rhs: Shape,
    device: B::Device,
}

impl<B: Backend, const D: usize> Benchmark for MatmulBenchmark<B, D> {
    type Input = (Tensor<B, D>, Tensor<B, D>);
    type Output = Tensor<B, D>;

    fn name(&self) -> String {
        format!("matmul-{:?}", B::FloatElem::dtype()).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        if self.shape_lhs == self.shape_rhs {
            vec![self.shape_lhs.dims.clone()]
        } else {
            vec![self.shape_lhs.dims.clone(), self.shape_rhs.dims.clone()]
        }
    }

    fn execute(&self, (lhs, rhs): Self::Input) -> Self::Output {
        lhs.matmul(rhs)
    }

    fn prepare(&self) -> Self::Input {
        let lhs = Tensor::random(self.shape_lhs.clone(), Distribution::Default, &self.device);
        let rhs = Tensor::random(self.shape_rhs.clone(), Distribution::Default, &self.device);

        (lhs, rhs)
    }

    fn sync(&self) {
        B::sync(&self.device)
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) -> Vec<BenchmarkResult> {
    [
        (2, 4096, 4096, 4096),
        (1, 6144, 6144, 6144),
        (4, 2048, 2048, 2048),
        (32, 1024, 1024, 1024),
        (256, 256, 256, 256),
    ]
    .into_iter()
    .map(|(b, m, n, k)| {
        let shape_lhs = [b, m, k].into();
        let shape_rhs = [b, k, n].into();

        MatmulBenchmark::<B, 3>::new(shape_lhs, shape_rhs, device.clone())
    })
    .map(run_benchmark)
    .collect()
}

fn main() {
    burnbench::bench_on_backend!();
}
