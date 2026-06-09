use burn::{
    module::Module,
    nn,
    tensor::{Device, Distribution, Tensor},
};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};

pub struct AutodiffOverheadBenchmark {
    config: nn::LstmConfig,
    lstm: nn::Lstm,
    device: Device,
}

impl Benchmark for AutodiffOverheadBenchmark {
    type Input = Tensor<3>;
    type Output = ();

    fn name(&self) -> String {
        format!("autodiff_overhead-{:?}", self.device.settings().float_dtype).to_lowercase()
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
        self.device.sync().unwrap();
    }
}

#[allow(dead_code)]
fn bench(device: &Device) -> Vec<BenchmarkResult> {
    let config = nn::LstmConfig::new(3, 3, true);
    // Autodiff is now a property of the device.
    let device = device.clone().autodiff();
    let lstm = config.init(&device);
    let benchmark = AutodiffOverheadBenchmark {
        lstm,
        config,
        device,
    };

    vec![run_benchmark(benchmark)]
}

fn main() {
    let device = backend_comparison::select_device();
    let results = bench(&device);
    backend_comparison::save(results, &device);
}
