use burn::tensor::{Device, Distribution, Shape, Tensor, activation::softmax};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};
use derive_new::new;

#[derive(new)]
struct SoftmaxBenchmark<const D: usize> {
    shape: Shape,
    dim: usize,
    device: Device,
}

impl<const D: usize> Benchmark for SoftmaxBenchmark<D> {
    type Input = Tensor<D>;
    type Output = Tensor<D>;

    fn name(&self) -> String {
        format!(
            "softmax-{:?}-{:?}",
            self.dim,
            self.device.settings().float_dtype
        )
        .to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }

    fn execute(&self, tensor: Self::Input) -> Self::Output {
        softmax(tensor, self.dim)
    }

    fn prepare(&self) -> Self::Input {
        Tensor::random(self.shape.clone(), Distribution::Default, &self.device)
    }

    fn sync(&self) {
        self.device.sync().unwrap();
    }
}

#[allow(dead_code)]
fn bench(device: &Device) -> Vec<BenchmarkResult> {
    [
        (2, 6144, 6144),
        (4, 4096, 4096),
        (8, 2048, 2048),
        (16, 1024, 1024),
        (256, 256, 256),
    ]
    .into_iter()
    .flat_map(|(a, b, c)| {
        let shape: Shape = [a, b, c].into();

        (0..shape.rank())
            .map(|dim| SoftmaxBenchmark::<3>::new(shape.clone(), dim, device.clone()))
            .collect::<Vec<_>>()
    })
    .map(run_benchmark)
    .collect()
}

fn main() {
    let device = backend_comparison::select_device();
    let results = bench(&device);
    backend_comparison::save(results, &device);
}
