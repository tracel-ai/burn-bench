use burn::tensor::{Distribution, Element, Shape, Tensor, TensorData, backend::Backend};
use burn_common::benchmark::{Benchmark, BenchmarkResult, run_benchmark};
use derive_new::new;

#[cfg(not(feature = "legacy-v16"))]
use rand::rng;
#[cfg(feature = "legacy-v16")]
use rand::thread_rng as rng;

#[derive(new)]
struct ToDataBenchmark<B: Backend, const D: usize> {
    shape: Shape,
    device: B::Device,
}

impl<B: Backend, const D: usize> Benchmark for ToDataBenchmark<B, D> {
    type Input = Tensor<B, D>;
    type Output = TensorData;

    fn name(&self) -> String {
        format!("to_data-{:?}", B::FloatElem::dtype()).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.dims.clone()]
    }

    fn execute(&self, args: Self::Input) -> Self::Output {
        args.to_data()
    }

    fn prepare(&self) -> Self::Input {
        Tensor::random(self.shape.clone(), Distribution::Default, &self.device)
    }

    fn sync(&self) {
        B::sync(&self.device)
    }
}

#[derive(new)]
struct FromDataBenchmark<B: Backend, const D: usize> {
    shape: Shape,
    device: B::Device,
}

impl<B: Backend, const D: usize> Benchmark for FromDataBenchmark<B, D> {
    type Input = (TensorData, B::Device);
    type Output = Tensor<B, D>;

    fn name(&self) -> String {
        "from_data".into()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.dims.clone()]
    }

    fn execute(&self, (data, device): Self::Input) -> Self::Output {
        Tensor::<B, D>::from_data(data.clone(), &device)
    }

    fn prepare(&self) -> Self::Input {
        (
            TensorData::random::<B::FloatElem, _, _>(
                self.shape.clone(),
                Distribution::Default,
                &mut rng(),
            ),
            self.device.clone(),
        )
    }

    fn sync(&self) {
        B::sync(&self.device)
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) -> Vec<BenchmarkResult> {
    const D: usize = 3;
    let shape: Shape = [32, 512, 1024].into();

    let to_benchmark = ToDataBenchmark::<B, D>::new(shape.clone(), device.clone());
    let from_benchmark = FromDataBenchmark::<B, D>::new(shape, device.clone());

    vec![run_benchmark(to_benchmark), run_benchmark(from_benchmark)]
}

fn main() {
    burnbench::bench_on_backend!();
}
