use burn::tensor::{Distribution, Element, Shape, Tensor, backend::Backend};
use burn_common::benchmark::{Benchmark, BenchmarkResult, run_benchmark};
use burnbench;

// Files retrieved during build to avoid reimplementing ResNet for benchmarks
mod block {
    extern crate alloc;
    include!(concat!(env!("OUT_DIR"), "/block.rs"));
}

mod model {
    include!(concat!(env!("OUT_DIR"), "/resnet.rs"));
}

pub struct ResNetBenchmark<B: Backend> {
    shape: Shape,
    device: B::Device,
}

impl<B: Backend> Benchmark for ResNetBenchmark<B> {
    type Args = (model::ResNet<B>, Tensor<B, 4>);

    fn name(&self) -> String {
        format!("resnet50-{:?}", B::FloatElem::dtype()).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.dims.clone()]
    }

    fn execute(&self, (model, input): Self::Args) {
        let _out = model.forward(input);
    }

    fn prepare(&self) -> Self::Args {
        // 1k classes like ImageNet
        let model = model::ResNet::resnet50(1000, &self.device);
        let input = Tensor::random(self.shape.clone(), Distribution::Default, &self.device);

        (model, input)
    }

    fn sync(&self) {
        B::sync(&self.device)
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) -> Vec<BenchmarkResult> {
    let benchmark = ResNetBenchmark::<B> {
        shape: [1, 3, 224, 224].into(),
        device: device.clone(),
    };

    vec![run_benchmark(benchmark)]
}

fn main() {
    burnbench::bench_on_backend!();
}
