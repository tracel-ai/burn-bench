use burn::tensor::{Device, Distribution, Float, Shape, Tensor};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};
use std::hint::black_box;

pub struct RandomBenchmark {
    shape: Shape,
    distribution: Distribution,
    device: Device,
}

impl Benchmark for RandomBenchmark {
    type Input = ();
    type Output = Tensor<3>;

    fn name(&self) -> String {
        format!(
            "random-{:?}-{:?}",
            self.distribution,
            self.device.settings().float_dtype,
        )
        .to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }

    fn execute(&self, (): Self::Input) -> Self::Output {
        Tensor::<3, Float>::random(self.shape.clone(), self.distribution, &self.device)
    }

    fn prepare(&self) -> Self::Input {}

    fn sync(&self) {
        self.device.sync().unwrap();
    }

    fn num_samples(&self) -> usize {
        40
    }
}

#[allow(dead_code)]
fn bench(device: &Device) -> Vec<BenchmarkResult> {
    let rand0 = RandomBenchmark {
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
    .map(|(batch_num, shape, distribution)| RandomBenchmark {
        shape: [batch_num, shape, shape].into(),
        distribution,
        device: device.clone(),
    })
    .map(run_benchmark)
    .collect()
}

fn main() {
    let device = backend_comparison::select_device();
    let results = bench(&device);
    backend_comparison::save(results, &device);
}
