use burn::tensor::Device;
use burn::{config::Config, module::Module, nn};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};
use derive_new::new;

#[derive(Module, Debug)]
struct BenchmarkModule {
    linears: Vec<nn::Linear>,
}

#[derive(Config, Debug)]
struct BenchmarkConfig {
    linear: nn::LinearConfig,
    num_layers: usize,
}

impl BenchmarkConfig {
    pub fn init(&self, device: &Device) -> BenchmarkModule {
        BenchmarkModule {
            linears: (0..self.num_layers)
                .map(|_| self.linear.init(device))
                .collect(),
        }
    }
    pub fn init_with(&self, record: BenchmarkModuleRecord) -> BenchmarkModule {
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
struct LoadRecordBenchmark {
    config: BenchmarkConfig,
    device: Device,
    kind: Kind,
}

impl Benchmark for LoadRecordBenchmark {
    type Input = BenchmarkModule;
    type Output = BenchmarkModule;

    fn name(&self) -> String {
        format!(
            "load_record_{:?}-{:?}",
            self.kind,
            self.device.settings().float_dtype
        )
        .to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![]
    }

    fn num_samples(&self) -> usize {
        10
    }

    fn execute(&self, module: Self::Input) -> Self::Output {
        let record = module.into_record();

        match self.kind {
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
        }
    }

    fn prepare(&self) -> Self::Input {
        let module = self.config.init(&self.device);
        // Force sync.

        module.clone()
    }

    fn sync(&self) {
        self.device.sync().unwrap();
    }
}

#[allow(dead_code)]
fn bench(device: &Device) -> Vec<BenchmarkResult> {
    let config = BenchmarkConfig::new(nn::LinearConfig::new(2048, 2048), 12);

    [Kind::Lazy, Kind::Sync, Kind::Manual]
        .into_iter()
        .map(|kind| LoadRecordBenchmark::new(config.clone(), device.clone(), kind))
        .map(run_benchmark)
        .collect()
}

fn main() {
    let device = backend_comparison::select_device();
    let results = bench(&device);
    backend_comparison::save(results, &device);
}
