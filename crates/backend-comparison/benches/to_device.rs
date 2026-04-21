use burn::{
    Tensor,
    prelude::Backend,
    tensor::{Distribution, Element, Shape},
};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};

pub struct ToDeviceBenchmark<B: Backend, const D: usize> {
    shape: Shape,
    device_src: B::Device,
    device_dst: B::Device,
}

impl<B: Backend, const D: usize> Benchmark for ToDeviceBenchmark<B, D> {
    type Input = Tensor<B, D>;
    type Output = Tensor<B, D>;

    fn name(&self) -> String {
        format!("to_device-{:?}", B::FloatElem::dtype()).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }

    fn execute(&self, input: Self::Input) -> Self::Output {
        input.to_device(&self.device_dst)
    }

    fn prepare(&self) -> Self::Input {
        Tensor::random(self.shape.clone(), Distribution::Default, &self.device_src)
    }

    fn sync(&self) {
        B::sync(&self.device_dst).unwrap()
    }

    fn num_samples(&self) -> usize {
        40
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(devices: &Vec<B::Device>) -> Vec<BenchmarkResult> {
    let conv1 = ToDeviceBenchmark::<B, 3> {
        shape: [32, 512, 1024].into(),
        device_src: devices[0].clone(),
        device_dst: devices[1].clone(),
    };

    let benches = vec![conv1];
    let mut results = Vec::new();

    for bench in benches {
        println!("Running {}", bench.name());
        let result = run_benchmark(bench);
        results.push(result);
    }

    results
}

fn main() {
    burnbench::bench_on_backend_multi_device!();
}
