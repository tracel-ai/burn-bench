use burn::{
    Tensor,
    prelude::Backend,
    tensor::{
        Distribution, Element, Shape, TensorPrimitive,
        backend::{
            DeviceOps,
            distributed::{DistributedBackend, ReduceOperation},
        },
    },
};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};

pub struct AllReduceBenchmark<B: Backend> {
    shape: Shape,
    devices: Vec<B::Device>,
}

impl<B: Backend + DistributedBackend> Benchmark for AllReduceBenchmark<B> {
    type Input = Vec<Tensor<B, 3>>;
    type Output = Vec<Tensor<B, 3>>;

    fn name(&self) -> String {
        format!("to_device-{:?}", B::FloatElem::dtype()).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }

    fn execute(&self, input: Self::Input) -> Self::Output {
        let mut out = vec![];
        let device_ids = self.devices.iter().map(|d| d.id()).collect::<Vec<_>>();
        for tensor in input {
            let result = B::all_reduce(
                tensor.into_primitive().tensor(),
                ReduceOperation::Sum,
                device_ids.clone(),
            );
            out.push(Tensor::new(TensorPrimitive::Float(result.resolve())));
        }
        out
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
            .for_each(|device| B::sync(&device).unwrap());
    }

    fn num_samples(&self) -> usize {
        40
    }
}

#[allow(dead_code)]
fn bench<B: Backend + DistributedBackend>(devices: &Vec<B::Device>) -> Vec<BenchmarkResult> {
    [[32, 512, 1024], [128, 512, 2048]]
        .into_iter()
        .map(|shape| AllReduceBenchmark::<B> {
            shape: shape.into(),
            devices: devices.clone(),
        })
        .map(run_benchmark)
        .collect()
}

fn main() {
    burnbench::bench_on_backend_multi_device!();
}
