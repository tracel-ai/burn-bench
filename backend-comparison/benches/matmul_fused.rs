use burn::tensor::{
    Distribution, Element, Shape, Tensor,
    activation::{gelu, relu},
    backend::Backend,
};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};
use derive_new::new;

#[derive(new)]
struct MatmulBenchmark<B: Backend, const D: usize> {
    shape_lhs: Shape,
    shape_rhs: Shape,
    device: B::Device,
}

impl<B: Backend, const D: usize> Benchmark for MatmulBenchmark<B, D> {
    type Input = (Tensor<B, D>, Tensor<B, D>, Tensor<B, 1>);
    type Output = Tensor<B, D>;

    fn name(&self) -> String {
        format!("matmul_relu_bias_gelu-{:?}", B::FloatElem::dtype()).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape_lhs.dims.clone(), self.shape_rhs.dims.clone()]
    }

    fn execute(&self, (lhs, rhs, bias): Self::Input) -> Self::Output {
        gelu(relu(lhs.matmul(rhs)) + bias.unsqueeze())
    }

    fn prepare(&self) -> Self::Input {
        let lhs = Tensor::random(self.shape_lhs.clone(), Distribution::Default, &self.device);
        let rhs = Tensor::random(self.shape_rhs.clone(), Distribution::Default, &self.device);
        let bias = Tensor::random(
            [self.shape_rhs.dims[2]],
            Distribution::Default,
            &self.device,
        );

        (lhs, rhs, bias)
    }

    fn sync(&self) {
        B::sync(&self.device)
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) -> Vec<BenchmarkResult> {
    [
        (2, 4096, 4096, 4096),
        (16, 2048, 2048, 2048),
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
