use burn::tensor::Device;
use burnbench::BenchmarkResult;

#[cfg(feature = "multi-device")]
mod distributed_benchmarks {
    use burn::tensor::{
        Device, Distribution, Shape, Tensor,
        distributed::{DistributedConfig, DistributedContext, ReduceOperation, all_reduce},
    };
    use burnbench::{Benchmark, BenchmarkResult, run_benchmark};

    pub struct AllReduceBenchmark {
        shape: Shape,
        devices: Vec<Device>,
    }

    impl Benchmark for AllReduceBenchmark {
        type Input = Vec<Tensor<3>>;
        type Output = Vec<Tensor<3>>;

        fn name(&self) -> String {
            format!("all_reduce-{:?}", self.devices[0].settings().float_dtype).to_lowercase()
        }

        fn shapes(&self) -> Vec<Vec<usize>> {
            vec![self.shape.to_vec()]
        }

        fn execute(&self, input: Self::Input) -> Self::Output {
            input
                .into_iter()
                .map(|tensor| {
                    all_reduce(tensor, ReduceOperation::Sum, self.devices.clone()).resolve()
                })
                .collect()
        }

        fn prepare(&self) -> Self::Input {
            self.devices
                .iter()
                .map(|device| Tensor::random(self.shape.clone(), Distribution::Default, device))
                .collect()
        }

        fn sync(&self) {
            self.devices
                .iter()
                .for_each(|device| device.sync().unwrap());
        }

        fn num_samples(&self) -> usize {
            40
        }
    }

    pub fn bench(devices: &[Device]) -> Vec<BenchmarkResult> {
        // Needs at least two devices to all-reduce across; skip gracefully on
        // hosts that expose fewer instead of panicking (e.g. on `devices[0]`).
        if devices.len() < 2 {
            eprintln!(
                "Skipping all_reduce benchmark: requires at least 2 devices, found {}.",
                devices.len()
            );
            return vec![];
        }
        // The distributed context starts the communication server for the
        // duration of the benchmark and tears it down on drop.
        let _ctx = DistributedContext::init(
            devices.to_vec(),
            DistributedConfig {
                all_reduce_op: ReduceOperation::Sum,
            },
        );

        [[32, 512, 1024], [128, 512, 2048]]
            .into_iter()
            .map(|shape| AllReduceBenchmark {
                shape: shape.into(),
                devices: devices.to_vec(),
            })
            .map(run_benchmark)
            .collect()
    }
}

#[cfg(feature = "multi-device")]
#[allow(dead_code)]
fn bench(devices: &[Device]) -> Vec<BenchmarkResult> {
    distributed_benchmarks::bench(devices)
}

#[cfg(not(feature = "multi-device"))]
#[allow(dead_code)]
fn bench(_devices: &[Device]) -> Vec<BenchmarkResult> {
    vec![]
}

fn main() {
    let devices = backend_comparison::select_devices();
    let results = bench(&devices);
    backend_comparison::save(results, &devices);
}
