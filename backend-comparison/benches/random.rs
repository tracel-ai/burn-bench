use burn::tensor::{Distribution, Element, Float, Shape, Tensor, backend::Backend};
use burn_common::benchmark::{Benchmark, BenchmarkResult, run_benchmark};
use std::hint::black_box;

pub struct RandomBenchmark<B: Backend> {
    shape: Shape,
    distribution: Distribution,
    device: B::Device,
}

impl<B: Backend> Benchmark for RandomBenchmark<B> {
    type Input = ();
    type Output = Tensor<B, 3>;

    fn name(&self) -> String {
        format!("random-{:?}-{:?}", self.distribution, B::FloatElem::dtype(),).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.dims.clone()]
    }

    fn execute(&self, (): Self::Input) -> Self::Output {
        Tensor::<B, 3, Float>::random(self.shape.clone(), self.distribution, &self.device)
    }

    fn prepare(&self) -> Self::Input {
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
        shape: [1, 512, 512, 512].into(),
        distribution: Distribution::Default,
        device: device.clone(),
    };

    let benches = vec![rand0];
    let mut results = Vec::new();

    for bench in benches {
        println!("Running {}", bench.name());
        let result = black_box(run_benchmark(bench));
        results.push(result);
    }

    [
        (1, 256, Distribution::Default),
        (1, 512, Distribution::Default),
        (1, 2048, Distribution::Default),
        (4, 512, Distribution::Default),
        (4, 2048, Distribution::Default),
        (16, 512, Distribution::Default),
        (16, 2048, Distribution::Default),
        (1, 512, Distribution::Bernoulli(0.45)),
        (1, 2048, Distribution::Bernoulli(0.45)),
        (1, 512, Distribution::Normal(10., 5.)),
        (1, 2048, Distribution::Normal(10., 5.)),
        (1, 512, Distribution::Uniform(5., 12.)),
        (1, 2048, Distribution::Uniform(5., 12.)),
    ]
    .into_iter()
    .map(|(batch_num, shape, distribution)| RandomBenchmark::<B> {
        shape: [batch_num, shape, shape].into(),
        distribution,
        device: device.clone(),
    })
    .map(run_benchmark)
    .collect()
}

fn main() {
    burnbench::bench_on_backend!();
}
