use burn::tensor::Device;
use burnbench::BenchmarkResult;

#[cfg(feature = "multi-device")]
mod to_device_benchmarks {
    use burn::tensor::{Device, Distribution, Shape, Tensor};
    use burnbench::{Benchmark, BenchmarkResult, run_benchmark};

    pub struct ToDeviceBenchmark {
        shape: Shape,
        device_src: Device,
        device_dst: Device,
    }

    impl Benchmark for ToDeviceBenchmark {
        type Input = Tensor<3>;
        type Output = Tensor<3>;

        fn name(&self) -> String {
            format!("to_device-{:?}", self.device_src.settings().float_dtype).to_lowercase()
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
            self.device_dst.sync().unwrap()
        }

        fn num_samples(&self) -> usize {
            40
        }
    }

    #[allow(dead_code)]
    pub fn bench(devices: &[Device]) -> Vec<BenchmarkResult> {
        // Needs at least two devices to move tensors between; skip gracefully on
        // hosts that expose fewer (e.g. a single GPU) instead of panicking.
        if devices.len() < 2 {
            eprintln!(
                "Skipping to_device benchmark: requires at least 2 devices, found {}.",
                devices.len()
            );
            return vec![];
        }
        [[32, 512, 1024], [128, 512, 2048]]
            .into_iter()
            .map(|shape| ToDeviceBenchmark {
                shape: shape.into(),
                device_src: devices[0].clone(),
                device_dst: devices[1].clone(),
            })
            .map(run_benchmark)
            .collect()
    }
}

#[cfg(feature = "multi-device")]
#[allow(dead_code)]
fn bench(devices: &[Device]) -> Vec<BenchmarkResult> {
    to_device_benchmarks::bench(devices)
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
