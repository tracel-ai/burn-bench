use burn::tensor::backend::Backend;
use burnbench;
use burnbench::BenchmarkResult;

#[cfg(all(feature = "distributed", feature = "multi-device"))]
mod distributed_benchmarks {
    use burn::{
        Tensor,
        collective::{AllReduceStrategy, CollectiveConfig, PeerId, ReduceOperation, all_reduce},
        prelude::Backend,
        tensor::{Distribution, Element, Shape, TensorPrimitive, backend::DeviceOps},
    };

    use burnbench::{Benchmark, BenchmarkResult, run_benchmark};

    pub struct AllReduceOldBenchmark<B: Backend> {
        shape: Shape,
        devices: Vec<B::Device>,
    }

    impl<B: Backend> Benchmark for AllReduceOldBenchmark<B> {
        type Input = (Vec<Tensor<B, 3>>, CollectiveConfig);
        type Output = Vec<Tensor<B, 3>>;

        fn name(&self) -> String {
            format!("all_reduce_old-{:?}", B::FloatElem::dtype()).to_lowercase()
        }

        fn shapes(&self) -> Vec<Vec<usize>> {
            vec![self.shape.to_vec()]
        }

        fn execute(&self, input: Self::Input) -> Self::Output {
            let mut out = vec![];
            let (input, config) = input.into();

            let mut recvs = vec![];

            for tensor in input {
                let (result_send, result_recv) = std::sync::mpsc::sync_channel::<Tensor<B, 3>>(1);
                recvs.push(result_recv);
                let config_cloned = config.clone();
                std::thread::spawn(move || {
                    let peer_id = PeerId::from(tensor.device().id().index_id);
                    burn::collective::register::<B>(peer_id, tensor.device(), config_cloned)
                        .unwrap();

                    let result = all_reduce::<B>(
                        peer_id,
                        tensor.into_primitive().tensor(),
                        ReduceOperation::Sum,
                    )
                    .unwrap();

                    result_send
                        .send(Tensor::new(TensorPrimitive::Float(result)))
                        .unwrap();
                });
            }

            for recv in recvs {
                let tensor = recv.recv().unwrap();
                out.push(tensor);
            }
            out
        }

        fn prepare(&self) -> Self::Input {
            let collective_config = CollectiveConfig::default()
                .with_num_devices(self.devices.len())
                .with_local_all_reduce_strategy(AllReduceStrategy::Tree(2));

            (
                self.devices
                    .iter()
                    .map(|device| Tensor::random(self.shape.clone(), Distribution::Default, device))
                    .collect(),
                collective_config,
            )
        }

        fn sync(&self) {
            self.devices
                .iter()
                .for_each(|device| B::sync(&device).unwrap());
        }

        fn num_samples(&self) -> usize {
            40
        }
    }

    pub fn bench<B: Backend>(devices: &Vec<B::Device>) -> Vec<BenchmarkResult> {
        [[32, 512, 1024], [128, 512, 2048]]
            .into_iter()
            .map(|shape| AllReduceOldBenchmark::<B> {
                shape: shape.into(),
                devices: devices.clone(),
            })
            .map(run_benchmark)
            .collect()
    }
}

#[cfg(all(feature = "distributed", feature = "multi-device"))]
#[allow(dead_code)]
fn bench<B: Backend>(devices: &Vec<B::Device>) -> Vec<BenchmarkResult> {
    distributed_benchmarks::bench::<B>(devices)
}

#[cfg(any(not(feature = "distributed"), not(feature = "multi-device")))]
#[allow(dead_code)]
fn bench<B: Backend>(_device: &B::Device) -> Vec<BenchmarkResult> {
    vec![]
}

fn main() {
    burnbench::bench_on_backend_multi_device!();
}
