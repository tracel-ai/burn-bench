use burn::tensor::{Device, Distribution, Shape, Tensor};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};
use rand::{
    RngExt as _, SeedableRng as _,
    rngs::{StdRng, SysRng},
};

pub struct BinaryBenchmark<const D: usize> {
    shape: Shape,
    device: Device,
}

impl<const D: usize> Benchmark for BinaryBenchmark<D> {
    type Input = (Tensor<D>, Tensor<D>);
    type Output = Tensor<D>;

    fn name(&self) -> String {
        format!("binary-{:?}", self.device.settings().float_dtype).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }

    fn execute(&self, (lhs, rhs): Self::Input) -> Self::Output {
        lhs.mul(rhs)
    }

    fn prepare(&self) -> Self::Input {
        let lhs = Tensor::<D>::random(self.shape.clone(), Distribution::Default, &self.device);
        let rhs = Tensor::<D>::random(self.shape.clone(), Distribution::Default, &self.device);

        (lhs, rhs)
    }

    fn sync(&self) {
        self.device.sync().unwrap();
    }
}

pub struct BinaryScalarBenchmark<const D: usize> {
    shape: Shape,
    device: Device,
}

impl<const D: usize> Benchmark for BinaryScalarBenchmark<D> {
    type Input = (Tensor<D>, f32);
    type Output = Tensor<D>;

    fn name(&self) -> String {
        format!("binary_scalar-{:?}", self.device.settings().float_dtype).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }

    fn execute(&self, (lhs, rhs): Self::Input) -> Self::Output {
        lhs.mul_scalar(rhs)
    }

    fn prepare(&self) -> Self::Input {
        let lhs = Tensor::random(self.shape.clone(), Distribution::Default, &self.device);
        let rhs = StdRng::try_from_rng(&mut SysRng).unwrap().random::<f32>();

        (lhs, rhs)
    }

    fn sync(&self) {
        self.device.sync().unwrap();
    }
}

#[allow(dead_code)]
fn bench(device: &Device) -> Vec<BenchmarkResult> {
    let benchmark = BinaryBenchmark::<3> {
        shape: [512, 512, 1024].into(),
        device: device.clone(),
    };
    let benchmark_scalar = BinaryScalarBenchmark::<3> {
        shape: [512, 512, 1024].into(),
        device: device.clone(),
    };

    vec![run_benchmark(benchmark), run_benchmark(benchmark_scalar)]
}

fn main() {
    let device = backend_comparison::select_device();
    let results = bench(&device);
    backend_comparison::save(results, &device);
}
