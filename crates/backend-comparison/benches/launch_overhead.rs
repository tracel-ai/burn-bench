use burn::tensor::{Device, Distribution, Shape, Tensor};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};

pub struct LaunchOverhead<const D: usize> {
    shape: Shape,
    device: Device,
    repetition: usize,
    num_threads: usize,
}

impl<const D: usize> Benchmark for LaunchOverhead<D> {
    type Input = (Tensor<D>, Tensor<D>);
    type Output = Tensor<D>;

    fn name(&self) -> String {
        format!(
            "launch-overhead-{:?}-reps-{}-threads-{}",
            self.device.settings().float_dtype,
            self.repetition,
            self.num_threads,
        )
        .to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }

    fn execute(&self, input: Self::Input) -> Self::Output {
        self.execute_inner(input)
    }

    fn prepare(&self) -> Self::Input {
        let lhs = Tensor::<D>::random(self.shape.clone(), Distribution::Default, &self.device);
        let rhs = Tensor::<D>::random(self.shape.clone(), Distribution::Default, &self.device);

        (lhs, rhs)
    }

    fn sync(&self) {
        self.device.sync().unwrap();
    }
}

impl<const D: usize> LaunchOverhead<D> {
    fn execute_inner(&self, (lhs, rhs): (Tensor<D>, Tensor<D>)) -> Tensor<D> {
        let mut handles = Vec::with_capacity(self.num_threads);

        enum Task<const D: usize> {
            Async(std::thread::JoinHandle<Tensor<D>>),
            Sync(Tensor<D>),
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
                    let new = Tensor::<D>::ones(shape.clone(), &device) * i as f32;

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
fn bench(device: &Device) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    for num_threads in [1, 4, 8, 16] {
        for shape in [[1, 4, 4, 4], [1, 8, 8, 8]] {
            for repetition in [512, 1024] {
                let benchmark = LaunchOverhead::<4> {
                    shape: shape.into(),
                    device: device.clone(),
                    repetition,
                    num_threads,
                };
                results.push(run_benchmark(benchmark));
            }
        }
    }

    results
}

fn main() {
    let device = backend_comparison::select_device();
    let results = bench(&device);
    backend_comparison::save(results, &device);
}
