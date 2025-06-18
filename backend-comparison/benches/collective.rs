use burn::tensor::backend::Backend;
use burnbench;
use burnbench::BenchmarkResult;

use std::{sync::mpsc::SyncSender, vec};

use burn::tensor::{Tensor};
use burnbench::Benchmark;
use burnbench::run_benchmark;

#[cfg(all(feature = "collective", feature = "distributed"))]
mod collective_remote_benchmarks {

    use super::*;
    use burn::{
        backend::{
            RemoteBackend,
            collective::{
                AggregateKind, AggregateParams, AggregateStrategy, all_reduce, register,
                reset_collective,
            },
            ir::BackendIr,
            remote::RemoteDevice,
        },
        tensor::Shape,
    };
    use std::marker::PhantomData;
    use tokio::runtime::Runtime;

    use derive_new::new;

    #[derive(Debug, Clone)]
    struct CollectiveBenchmarkInput<const D: usize> {
        input_data: Vec<Tensor<RemoteBackend, D>>,
    }

    #[derive(new)]
    struct CollectiveBenchmark<const D: usize> {
        shape: Shape,
        params: AggregateParams,
        devices: Vec<RemoteDevice>,
    }

    struct LocalServer<B: BackendIr> {
        port: u16,
        _runtime: Runtime,
        _phantom_data: PhantomData<B>,
    }

    impl<B: BackendIr> LocalServer<B> {
        pub fn new(port: u16) -> Self {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_io()
                .enable_time()
                .build()
                .unwrap();

            runtime.spawn(burn::server::start_async::<B>(Default::default(), port));

            Self {
                port,
                _runtime: runtime,
                _phantom_data: Default::default(),
            }
        }

        pub fn get_device(&self) -> RemoteDevice {
            RemoteDevice::new(&format!("ws://localhost:{}", self.port))
        }
    }

    pub fn run_peer<const D: usize>(
        id: u32,
        peer_count: u32,
        params: AggregateParams,
        input: Tensor<RemoteBackend, D>,
        output: SyncSender<Tensor<RemoteBackend, D>>,
    ) {
        register::<RemoteBackend>(id, peer_count);

        let tensor = all_reduce(input, params);

        output.send(tensor).unwrap();
    }

    impl<const D: usize> Benchmark for CollectiveBenchmark<D> {
        type Input = CollectiveBenchmarkInput<D>;
        type Output = Result<(), ()>;

        fn name(&self) -> String {
            format!(
                "collective-{:?}-{:?}-{:?}",
                self.devices.len(),
                self.params.kind,
                self.params.strategy
            )
            .to_lowercase()
        }

        fn shapes(&self) -> Vec<Vec<usize>> {
            vec![self.shape.dims.clone()]
        }

        fn execute(&self, mut input: Self::Input) -> Self::Output {
            reset_collective::<RemoteBackend>();

            let (send, recv) = std::sync::mpsc::sync_channel(1);

            let peer_count = self.devices.len() as u32;
            for (id, tensor) in input.input_data.drain(..).enumerate() {
                let send = send.clone();
                let params = self.params.clone();
                let input = tensor;
                std::thread::spawn(move || {
                    run_peer::<D>(id as u32, peer_count as u32, params, input, send)
                });
            }

            for _ in 0..peer_count {
                let _result = recv.recv().map_err(|_| ())?;
            }

            Ok(())
        }

        fn prepare(&self) -> Self::Input {
            let input_data: Vec<Tensor<RemoteBackend, D>> = self
                .devices
                .iter()
                .map(|dev| {
                    Tensor::random(self.shape.clone(), burn::tensor::Distribution::Default, dev)
                })
                .collect();

            CollectiveBenchmarkInput { input_data }
        }

        fn sync(&self) {
            self.devices.iter().for_each(|dev| RemoteBackend::sync(dev));
        }
    }

    pub fn bench<B: BackendIr>(_device: &B::Device) -> Vec<BenchmarkResult> {

        let peer_count = 2;

        let mut servers = vec![];
        let mut devices = vec![];
        for i in 0..peer_count {
            let port = 3000 + i;
            let server = LocalServer::<B>::new(port);
            devices.push(server.get_device());
            servers.push(server);
        }

        let shapes = vec![
            vec![1, 1, 8],
            // vec![16, 16, 16],
            // vec![32, 64, 128],
            // vec![64, 128, 256],
        ];

        let kinds = vec![AggregateKind::Sum, AggregateKind::Mean];

        let strategies = vec![
            AggregateStrategy::Centralized,
            // AggregateStrategy::Tree(2),
            // AggregateStrategy::Tree(5),
            // AggregateStrategy::Ring,
        ];

        let mut results = Vec::new();

        for shape_dims in shapes {
            for kind in &kinds {
                for strategy in &strategies {
                    let benchmark: CollectiveBenchmark<3> = CollectiveBenchmark {
                        shape: Shape {
                            dims: shape_dims.clone(),
                        },
                        params: AggregateParams {
                            kind: kind.clone(),
                            strategy: strategy.clone(),
                        },
                        devices: devices.clone(),
                    };

                    results.push(run_benchmark(benchmark));
                }
            }
        }

        results
    }
}

#[cfg(feature = "collective")]
#[allow(dead_code)]
fn bench<B: burn::backend::ir::BackendIr>(device: &B::Device) -> Vec<BenchmarkResult> {
    collective_remote_benchmarks::bench::<B>(device)
}

#[cfg(not(feature = "collective"))]
#[allow(dead_code)]
fn bench<B: Backend>(_device: &B::Device) -> Vec<BenchmarkResult> {
    vec![]
}

fn main() {
    burnbench::bench_on_backend!();
}
