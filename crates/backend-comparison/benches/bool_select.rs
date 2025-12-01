use burn::tensor::{Bool, Int, Shape, Tensor, TensorData, backend::Backend};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};
use derive_new::new;
use rand::Rng;

#[cfg(not(feature = "legacy-v16"))]
use rand::rng;
#[cfg(feature = "legacy-v16")]
use rand::thread_rng as rng;

#[derive(new)]
struct BoolSelectBenchmark<B: Backend, const D: usize> {
    shape: Shape,
    dim: usize,
    indices_count: usize,
    device: B::Device,
}

impl<B: Backend, const D: usize> Benchmark for BoolSelectBenchmark<B, D> {
    type Input = (Tensor<B, D, Bool>, Tensor<B, 1, Int>);
    type Output = Tensor<B, D, Bool>;

    fn name(&self) -> String {
        format!("bool_select_dim{}", self.dim)
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.dims.clone(), vec![self.indices_count]]
    }

    fn execute(&self, (tensor, indices): Self::Input) -> Self::Output {
        tensor.select(self.dim, indices)
    }

    fn prepare(&self) -> Self::Input {
        // Create boolean tensor using TensorData
        let bool_data: Vec<bool> = (0..self.shape.num_elements())
            .map(|_| rng().random_bool(0.5))
            .collect();
        let tensor_data = TensorData::new(bool_data, self.shape.clone());
        let tensor = Tensor::<B, D, Bool>::from_data(tensor_data, &self.device);

        // Generate valid random indices for the specified dimension
        let max_index = self.shape.dims[self.dim];
        let indices_data: Vec<i32> = (0..self.indices_count)
            .map(|_| rng().random_range(0..max_index) as i32)
            .collect();
        let indices_tensor_data = TensorData::new(indices_data, [self.indices_count]);
        let indices = Tensor::<B, 1, Int>::from_data(indices_tensor_data, &self.device);

        (tensor, indices)
    }

    fn sync(&self) {
        B::sync(&self.device).unwrap();
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    // Test configurations: (shape, dim, indices_count)
    let test_configs: Vec<(Shape, usize, usize)> = vec![
        // Small tensor
        ([32, 32, 32].into(), 0, 8),
        // Medium tensor
        ([64, 128, 256].into(), 1, 16),
        // Large tensor
        ([128, 256, 512].into(), 2, 32),
    ];

    for (shape, dim, indices_count) in test_configs {
        let benchmark = BoolSelectBenchmark::<B, 3>::new(shape, dim, indices_count, device.clone());
        results.push(run_benchmark(benchmark));
    }

    results
}

fn main() {
    burnbench::bench_on_backend!();
}
