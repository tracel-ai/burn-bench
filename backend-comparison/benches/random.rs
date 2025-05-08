use burn::tensor::{backend::Backend, Distribution, Element, Float, Shape, Tensor};
use burn_common::benchmark::{Benchmark, BenchmarkResult, run_benchmark};
use std::hint::black_box;

pub struct RandomBenchmark<B: Backend> {
    suffix: &'static str,
    shape: Shape,
    device: B::Device,
}

impl<B: Backend> Benchmark for RandomBenchmark<B> {
    type Args = ();

    fn name(&self) -> String {
        format!("random-default-{}-{:?}", self.suffix, B::FloatElem::dtype()).to_lowercase()
    }

    fn execute(&self, (): Self::Args) {
        Tensor::<B, 4, Float>::random(self.shape.clone(), Distribution::Default, &self.device);
    }

    fn prepare(&self) -> Self::Args {
        ()
    }

    fn sync(&self) {
        B::sync(&self.device)
    }

    fn num_samples(&self) -> usize {
        40
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) -> Vec<BenchmarkResult> {
    let rand0 = RandomBenchmark::<B> {
        suffix: "rand_128x128",
        shape: [1, 512, 512, 512].into(),
        device: device.clone(),
    };

    let benches = vec![
        rand0
    ];
    let mut results = Vec::new();

    for bench in benches {
        println!("Running {}", bench.name());
        let result = black_box(run_benchmark(bench));
        results.push(result);
    }

    results
}

fn main() {
    burnbench::bench_on_backend!();
}
