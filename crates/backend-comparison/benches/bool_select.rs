use burn::tensor::{Bool, Device, Int, Shape, Tensor, TensorData};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};
use derive_new::new;
use rand::{RngExt as _, rng};

#[derive(new)]
struct BoolSelectBenchmark<const D: usize> {
    shape: Shape,
    dim: usize,
    indices_count: usize,
    device: Device,
}

impl<const D: usize> Benchmark for BoolSelectBenchmark<D> {
    type Input = (Tensor<D, Bool>, Tensor<1, Int>);
    type Output = Tensor<D, Bool>;

    fn name(&self) -> String {
        format!("bool_select_dim{}", self.dim)
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec(), vec![self.indices_count]]
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
        let tensor = Tensor::<D, Bool>::from_data(tensor_data, &self.device);

        // Generate valid random indices for the specified dimension
        let max_index = self.shape[self.dim];
        let indices_data: Vec<i32> = (0..self.indices_count)
            .map(|_| rng().random_range(0..max_index) as i32)
            .collect();
        let indices_tensor_data = TensorData::new(indices_data, [self.indices_count]);
        let indices = Tensor::<1, Int>::from_data(indices_tensor_data, &self.device);

        (tensor, indices)
    }

    fn sync(&self) {
        self.device.sync().unwrap();
    }
}

#[allow(dead_code)]
fn bench(device: &Device) -> Vec<BenchmarkResult> {
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
        let benchmark = BoolSelectBenchmark::<3>::new(shape, dim, indices_count, device.clone());
        results.push(run_benchmark(benchmark));
    }

    results
}

fn main() {
    let device = backend_comparison::select_device();
    let results = bench(&device);
    backend_comparison::save(results, &device);
}
