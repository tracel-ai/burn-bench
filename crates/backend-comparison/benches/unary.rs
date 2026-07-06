use burn::tensor::{Device, Distribution, Shape, Tensor};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};
use derive_new::new;

#[derive(new)]
struct UnaryBenchmark<const D: usize> {
    shape: Shape,
    device: Device,
}

impl<const D: usize> Benchmark for UnaryBenchmark<D> {
    type Input = Tensor<D>;
    type Output = Tensor<D>;

    fn name(&self) -> String {
        format!("unary-{:?}", self.device.settings().float_dtype).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }

    fn total_bytes(&self) -> Option<u64> {
        // Elementwise unary: read every element once, write every element once.
        let elems: u64 = self.shape.to_vec().iter().map(|&d| d as u64).product();
        let size = backend_comparison::dtype_size(self.device.settings().float_dtype) as u64;
        Some(2 * elems * size)
    }

    fn execute(&self, args: Self::Input) -> Self::Output {
        // Choice of tanh is arbitrary
        args.tanh()
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
    const D: usize = 3;
    let shape: Shape = [32, 512, 1024].into();

    let benchmark = UnaryBenchmark::<D>::new(shape, device.clone());

    vec![run_benchmark(benchmark)]
}

fn main() {
    let device = backend_comparison::select_device();
    let results = bench(&device);
    backend_comparison::save(results, &device);
}
