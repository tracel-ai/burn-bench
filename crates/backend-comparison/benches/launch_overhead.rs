use burn::tensor::{Distribution, Element, Shape, Tensor, backend::Backend};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};

pub struct LaunchOverhead<B: Backend, const D: usize> {
    shape: Shape,
    device: B::Device,
    repetition: usize,
    num_threads: usize,
    scoped: bool,
}

impl<B: Backend, const D: usize> Benchmark for LaunchOverhead<B, D> {
    type Input = (Tensor<B, D>, Tensor<B, D>);
    type Output = Tensor<B, D>;

    fn name(&self) -> String {
        format!(
            "launch-overhead-{:?}-reps-{}-threads-{}-{}",
            B::FloatElem::dtype(),
            self.repetition,
            self.num_threads,
            self.scoped,
        )
        .to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }

    fn execute(&self, input: Self::Input) -> Self::Output {
        if self.scoped {
            // We use memory_persistent_allocations since it put all the code executing on the
            // server's thread.
            B::memory_persistent_allocations(&self.device, input, |input| self.execute_inner(input))
        } else {
            self.execute_inner(input)
        }
    }

    fn prepare(&self) -> Self::Input {
        let lhs = Tensor::<B, D>::random(self.shape.clone(), Distribution::Default, &self.device);
        let rhs = Tensor::<B, D>::random(self.shape.clone(), Distribution::Default, &self.device);

        (lhs, rhs)
    }

    fn sync(&self) {
        B::sync(&self.device).unwrap();
    }
}

impl<B: Backend, const D: usize> LaunchOverhead<B, D> {
    fn execute_inner(&self, (lhs, rhs): (Tensor<B, D>, Tensor<B, D>)) -> Tensor<B, D> {
        let mut handles = Vec::with_capacity(self.num_threads);

        enum Task<B: Backend, const D: usize> {
            Async(std::thread::JoinHandle<Tensor<B, D>>),
            Sync(Tensor<B, D>),
        }

        for _ in 0..self.num_threads {
            let lhs = lhs.clone();
            let rhs = rhs.clone();
            let repetition = self.repetition;
            let shape = self.shape.clone();
            let device = self.device.clone();

            let func = move || {
                let mut tmp = lhs.clone();
                for i in 0..repetition {
                    // let new = Tensor::<B, D>::random(shape.clone(), Distribution::Default, &device);
                    let new = tmp.clone().log();

                    if i % 2 == 0 {
                        tmp = tmp.clone().mul(rhs.clone()) + new;
                    } else {
                        tmp = lhs.clone().add(tmp.clone()) + new;
                    }
                }

                tmp
            };
            if self.num_threads > 1 {
                let handle = std::thread::spawn(func);
                handles.push(Task::Async(handle));
            } else {
                let tmp = func();
                handles.push(Task::Sync(tmp));
            }
        }

        let mut tensors = Vec::with_capacity(self.num_threads);
        for handle in handles {
            let tensor = match handle {
                Task::Async(join_handle) => join_handle.join().unwrap(),
                Task::Sync(tensor) => tensor,
            };
            tensors.push(tensor);
        }

        Tensor::cat(tensors, 0)
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    for num_threads in [1, 4] {
        // for shape in [[1, 8, 8, 8], [1, 32, 32, 32], [1, 128, 128, 128]] {
        for shape in [[1, 8, 8, 8]] {
            for repetition in [1024] {
                let benchmark = LaunchOverhead::<B, 4> {
                    shape: shape.into(),
                    device: device.clone(),
                    repetition,
                    num_threads,
                    scoped: false,
                };
                results.push(run_benchmark(benchmark));
            }
        }
    }

    results
}

fn main() {
    burnbench::bench_on_backend!();
}
