use burn::tensor::{Shape, backend::Backend};
use burnbench;
use burnbench::BenchmarkResult;
use derive_new::new;

use std::{sync::mpsc::SyncSender, vec};

use burn::tensor::{Element, Tensor};
use burnbench::Benchmark;
use burnbench::run_benchmark;

#[cfg(all(
    feature = "collective",
    not(feature = "legacy-v16"),
    not(feature = "legacy-v17")
))]
mod collective_benchmarks {
    use super::*;

    use burn::backend::collective::{all_reduce, register, reset_collective, AggregateKind, AggregateParams, AggregateStrategy};

    #[derive(Debug, Clone)]
    struct CollectiveBenchmarkInput<B: Backend, const D: usize> {
        input_data: Vec<Tensor<B, D>>,
    }

    #[derive(new)]
    struct CollectiveBenchmark<B: Backend, const D: usize> {
        shape: Shape,
        params: AggregateParams,
        devices: Vec<B::Device>,
    }

    pub fn run_peer<B: Backend, const D: usize>(
        id: u32,
        peer_count: u32,
        params: AggregateParams,
        input: Tensor<B, D>,
        output: SyncSender<Tensor<B, D>>,
    ) {
        register::<B>(id, peer_count);

        let tensor = all_reduce(input, params);

        output.send(tensor).unwrap();
    }

    impl<B: Backend, const D: usize> Benchmark for CollectiveBenchmark<B, D> {
        type Input = CollectiveBenchmarkInput<B, D>;
        type Output = Result<(), ()>;

        fn name(&self) -> String {
            format!(
                "collective-{:?}-{:?}-{:?}-x{:?}",
                self.devices.len(),
                self.params.kind,
                self.params.strategy,
                B::FloatElem::dtype()
            )
            .to_lowercase()
        }

        fn shapes(&self) -> Vec<Vec<usize>> {
            vec![self.shape.dims.clone()]
        }

        fn execute(&self, mut input: Self::Input) -> Self::Output {
            reset_collective::<B>();

            let (send, recv) = std::sync::mpsc::sync_channel(1);

            let peer_count = self.devices.len() as u32;
            for (id, tensor) in input.input_data.drain(..).enumerate() {
                let send = send.clone();
                let params = self.params.clone();
                let input = tensor;
                std::thread::spawn(move || {
                    run_peer::<B, D>(id as u32, peer_count as u32, params, input, send)
                });
            }

            for _ in 0..peer_count {
                let _result = recv.recv()
                    .map_err(|_| ())?;
            }

            Ok(())
        }

        fn prepare(&self) -> Self::Input {
            let input_data: Vec<Tensor<B, D>> = self
                .devices
                .iter()
                .map(|dev| {
                    Tensor::random(self.shape.clone(), burn::tensor::Distribution::Default, dev)
                })
                .collect();

            CollectiveBenchmarkInput { input_data }
        }

        fn sync(&self) {
            self.devices.iter().for_each(|dev| B::sync(dev));
        }
    }

    // pub fn bench<B: Backend>(device: &B::Device) -> Vec<BenchmarkResult> {
        
    //     let benchmark: CollectiveBenchmark<B, 3> = CollectiveBenchmark {
    //         shape: Shape {
    //             dims: vec![8, 8, 8],
    //         },
    //         params: AggregateParams {
    //             kind: AggregateKind::Mean,
    //             strategy: AggregateStrategy::Ring,
    //         },
    //         devices: vec![device.clone(); 4],
    //     };

    //     vec![run_benchmark(benchmark)]
    // }

    pub fn bench<B: Backend>(device: &B::Device) -> Vec<BenchmarkResult> {
        use AggregateKind::*;
        use AggregateStrategy::*;
    
        let shapes = vec![
            vec![8, 8, 8],
            vec![16, 16, 16],
            vec![32, 64, 128],
            vec![64, 128, 256],
        ];
    
        let kinds = vec![
            Sum,
            Mean,
        ];
    
        let strategies = vec![
            Ring,
            Tree(2),
            Tree(5),
            Centralized,
        ];
    
        let peer_count = 4;

        let mut results = Vec::new();
    
        for shape_dims in shapes {
            for kind in &kinds {
                for strategy in &strategies {
                    let benchmark: CollectiveBenchmark<B, 3> = CollectiveBenchmark {
                        shape: Shape {
                            dims: shape_dims.clone(),
                        },
                        params: AggregateParams {
                            kind: kind.clone(),
                            strategy: strategy.clone(),
                        },
                        devices: vec![device.clone(); peer_count],
                    };
    
                    results.push(run_benchmark(benchmark));
                }
            }
        }
    
        results
    }
}

#[cfg(all(
    feature = "collective",
    not(feature = "legacy-v16"),
    not(feature = "legacy-v17")
))]
#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) -> Vec<BenchmarkResult> {
    collective_benchmarks::bench::<B>(device)
}

#[cfg(any(
    not(feature = "collective"),
    feature = "legacy-v16",
    feature = "legacy-v17"
))]
#[allow(dead_code)]
fn bench<B: Backend>(_device: &B::Device) -> Vec<BenchmarkResult> {
    vec![]
}

fn main() {
    burnbench::bench_on_backend!();
}
