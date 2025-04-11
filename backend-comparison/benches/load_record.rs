use burn::tensor::backend::Backend;
use burn::tensor::{Device, Element};
use burn::{config::Config, module::Module, nn};
use burn_common::benchmark::{Benchmark, BenchmarkResult, run_benchmark};
use derive_new::new;

#[derive(Module, Debug)]
struct BenchmarkModule<B: Backend> {
    linears: Vec<nn::Linear<B>>,
}

#[derive(Config, Debug)]
struct BenchmarkConfig {
    linear: nn::LinearConfig,
    num_layers: usize,
}

impl BenchmarkConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BenchmarkModule<B> {
        BenchmarkModule {
            linears: (0..self.num_layers)
                .map(|_| self.linear.init(device))
                .collect(),
        }
    }
    pub fn init_with<B: Backend>(&self, record: BenchmarkModuleRecord<B>) -> BenchmarkModule<B> {
        BenchmarkModule {
            linears: record
                .linears
                .into_iter()
                .map(|record| nn::Linear {
                    weight: record.weight,
                    bias: record.bias,
                })
                .collect(),
        }
    }
}

#[derive(Debug)]
enum Kind {
    Lazy,
    Sync,
    Manual,
}

#[derive(new)]
struct LoadRecordBenchmark<B: Backend> {
    config: BenchmarkConfig,
    device: Device<B>,
    kind: Kind,
}

impl<B: Backend> Benchmark for LoadRecordBenchmark<B> {
    type Args = BenchmarkModule<B>;

    fn name(&self) -> String {
        format!("load_record_{:?}-{:?}", self.kind, B::FloatElem::dtype()).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![]
    }

    fn num_samples(&self) -> usize {
        10
    }

    fn execute(&self, module: Self::Args) {
        let record = module.into_record();

        let _ = match self.kind {
            Kind::Lazy => {
                let module = self.config.init(&self.device);
                module.load_record(record)
            }
            Kind::Sync => {
                let module = self.config.init(&self.device);
                // Force sync.
                let _ = module.clone();
                module.load_record(record)
            }
            Kind::Manual => self.config.init_with(record),
        };
    }

    fn prepare(&self) -> Self::Args {
        let module = self.config.init(&self.device);
        // Force sync.

        module.clone()
    }

    fn sync(&self) {
        B::sync(&self.device)
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) -> Vec<BenchmarkResult> {
    let config = BenchmarkConfig::new(nn::LinearConfig::new(2048, 2048), 12);

    [Kind::Lazy, Kind::Sync, Kind::Manual]
        .into_iter()
        .map(|kind| LoadRecordBenchmark::<B>::new(config.clone(), device.clone(), kind))
        .map(run_benchmark)
        .collect()
}

fn main() {
    burnbench::bench_on_backend!();
}
