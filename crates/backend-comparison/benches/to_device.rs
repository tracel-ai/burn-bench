use burn::tensor::backend::Backend;
use burnbench;
use burnbench::BenchmarkResult;

#[cfg(feature = "multi-device")]
mod to_device_benchmarks {
    use burn::{
        Tensor,
        prelude::Backend,
        tensor::{Distribution, Element, Shape},
    };
    use burnbench::{Benchmark, BenchmarkResult, run_benchmark};

    pub struct ToDeviceBenchmark<B: Backend> {
        shape: Shape,
        device_src: B::Device,
        device_dst: B::Device,
    }

    impl<B: Backend> Benchmark for ToDeviceBenchmark<B> {
        type Input = Tensor<B, 3>;
        type Output = Tensor<B, 3>;

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
    pub fn bench<B: Backend>(devices: &Vec<B::Device>) -> Vec<BenchmarkResult> {
        assert!(devices.len() >= 2);
        [[32, 512, 1024], [128, 512, 2048]]
            .into_iter()
            .map(|shape| ToDeviceBenchmark::<B> {
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
fn bench<B: Backend>(devices: &Vec<B::Device>) -> Vec<BenchmarkResult> {
    to_device_benchmarks::bench::<B>(devices)
}

#[cfg(not(feature = "multi-device"))]
#[allow(dead_code)]
fn bench<B: Backend>(_device: &B::Device) -> Vec<BenchmarkResult> {
    vec![]
}

fn main() {
    burnbench::bench_on_backend_multi_device!();
}
