use burn::tensor::{Distribution, Element, Shape, Tensor, backend::Backend};
use burn_common::benchmark::{Benchmark, BenchmarkResult, run_benchmark};
use std::marker::PhantomData;

#[cfg(not(feature = "legacy-v16"))]
use rand::rng;
#[cfg(feature = "legacy-v16")]
use rand::thread_rng as rng;

pub struct BinaryBenchmark<B: Backend, const D: usize> {
    shape: Shape,
    device: B::Device,
}

impl<B: Backend, const D: usize> Benchmark for BinaryBenchmark<B, D> {
    type Args = (Tensor<B, D>, Tensor<B, D>);

    fn name(&self) -> String {
        format!("binary-{:?}", B::FloatElem::dtype()).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.dims.clone()]
    }

    fn execute(&self, (lhs, rhs): Self::Args) {
        let _ = lhs.greater(rhs);
    }

    fn prepare(&self) -> Self::Args {
        let lhs = Tensor::<B, D>::random(self.shape.clone(), Distribution::Default, &self.device);
        let rhs = Tensor::<B, D>::random(self.shape.clone(), Distribution::Default, &self.device);

        (lhs, rhs)
    }

    fn sync(&self) {
        B::sync(&self.device);
    }
}

pub struct BinaryScalarBenchmark<B: Backend, const D: usize, E: Element> {
    shape: Shape,
    device: B::Device,
    _ty: PhantomData<E>,
}

impl<B: Backend, const D: usize, E: Element> Benchmark for BinaryScalarBenchmark<B, D, E> {
    type Args = (Tensor<B, D>, E);

    fn name(&self) -> String {
        format!("binary_scalar-{:?}", B::FloatElem::dtype()).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.dims.clone()]
    }

    fn execute(&self, (lhs, rhs): Self::Args) {
        let _ = lhs.equal_elem(rhs);
    }

    fn prepare(&self) -> Self::Args {
        let lhs = Tensor::random(self.shape.clone(), Distribution::Default, &self.device);
        let rhs = E::random(Distribution::Default, &mut rng());

        (lhs, rhs)
    }

    fn sync(&self) {
        B::sync(&self.device);
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) -> Vec<BenchmarkResult> {
    let benchmark = BinaryBenchmark::<B, 3> {
        shape: [512, 512, 1024].into(),
        device: device.clone(),
    };
    let benchmark_scalar = BinaryScalarBenchmark::<B, 3, B::FloatElem> {
        shape: [512, 512, 1024].into(),
        device: device.clone(),
        _ty: PhantomData,
    };

    vec![run_benchmark(benchmark), run_benchmark(benchmark_scalar)]
}

fn main() {
    burnbench::bench_on_backend!();
}
