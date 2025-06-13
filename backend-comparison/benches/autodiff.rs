use burn::{
    module::Module,
    nn,
    tensor::{
        Distribution, Element, Tensor,
        backend::{AutodiffBackend, Backend},
    },
};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};

pub struct AutodiffOverheadBenchmark<B: AutodiffBackend> {
    config: nn::LstmConfig,
    lstm: nn::Lstm<B>,
    device: B::Device,
}

impl<B: AutodiffBackend> Benchmark for AutodiffOverheadBenchmark<B> {
    type Input = Tensor<B, 3>;
    type Output = ();

    fn name(&self) -> String {
        format!("autodiff_overhead-{:?}", B::FloatElem::dtype()).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![]
    }

    fn execute(&self, input: Self::Input) -> Self::Output {
        for _ in 0..20 {
            let input = input.clone().detach();
            let mut cell = input.clone();
            let lstm = self.lstm.clone().fork(&input.device());

            for _ in 0..10 {
                let (cells, _) = lstm.forward(input.clone(), None);
                cell = cell + cells;
            }

            let _grads = cell.backward();
        }
    }

    fn prepare(&self) -> Self::Input {
        let shape = [1, 3, self.config.d_hidden];
        Tensor::random(shape, Distribution::Default, &self.device)
    }

    fn sync(&self) {
        B::sync(&self.device)
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) -> Vec<BenchmarkResult> {
    let config = nn::LstmConfig::new(3, 3, true);
    let lstm = config.init(device);
    let benchmark = AutodiffOverheadBenchmark::<burn::backend::Autodiff<B>> {
        lstm,
        config,
        device: device.clone(),
    };

    vec![run_benchmark(benchmark)]
}

fn main() {
    burnbench::bench_on_backend!();
}
