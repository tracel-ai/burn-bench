use burn::tensor::{
    Device, Distribution, Shape, Tensor,
    activation::{gelu, relu},
};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};
use derive_new::new;

#[derive(new)]
struct MatmulBenchmark<const D: usize> {
    shape_lhs: Shape,
    shape_rhs: Shape,
    device: Device,
}

impl<const D: usize> Benchmark for MatmulBenchmark<D> {
    type Input = (Tensor<D>, Tensor<D>, Tensor<1>);
    type Output = Tensor<D>;

    fn name(&self) -> String {
        format!(
            "matmul_relu_bias_gelu-{:?}",
            self.device.settings().float_dtype
        )
        .to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape_lhs.to_vec(), self.shape_rhs.to_vec()]
    }

    fn execute(&self, (lhs, rhs, bias): Self::Input) -> Self::Output {
        gelu(relu(lhs.matmul(rhs)) + bias.unsqueeze())
    }

    fn prepare(&self) -> Self::Input {
        let lhs = Tensor::random(self.shape_lhs.clone(), Distribution::Default, &self.device);
        let rhs = Tensor::random(self.shape_rhs.clone(), Distribution::Default, &self.device);
        let bias = Tensor::random([self.shape_rhs[2]], Distribution::Default, &self.device);

        (lhs, rhs, bias)
    }

    fn sync(&self) {
        self.device.sync().unwrap();
    }
}

#[allow(dead_code)]
fn bench(device: &Device) -> Vec<BenchmarkResult> {
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

        MatmulBenchmark::<3>::new(shape_lhs, shape_rhs, device.clone())
    })
    .map(run_benchmark)
    .collect()
}

fn main() {
    let device = backend_comparison::select_device();
    let results = bench(&device);
    backend_comparison::save(results, &device);
}
