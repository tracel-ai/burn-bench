use burn::{
    nn::{self, LinearConfig, LinearLayout},
    tensor::{Distribution, Element, Shape, Tensor, backend::Backend},
};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};

pub struct LinearBench<B: Backend> {
    shape: Shape,
    linear: nn::Linear<B>,
    device: B::Device,
    name: String,
}

impl<B: Backend> Benchmark for LinearBench<B> {
    type Input = (nn::Linear<B>, Tensor<B, 4>);
    type Output = Tensor<B, 4>;

    fn name(&self) -> String {
        format!("linear-{}-{:?}", self.name, B::FloatElem::dtype()).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        let weight = self.linear.weight.val().shape();
        vec![self.shape.dims.clone(), weight.dims]
    }

    fn execute(&self, (linear, signal): Self::Input) -> Self::Output {
        linear.forward(signal)
    }

    fn prepare(&self) -> Self::Input {
        let signal = Tensor::random(self.shape.clone(), Distribution::Default, &self.device);
        let linear = self.linear.clone();
        (linear, signal)
    }

    fn sync(&self) {
        B::sync(&self.device)
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();
    for layout in [LinearLayout::Row, LinearLayout::Col] {
        for bias in [true, false] {
            for b in [1, 256] {
                for d_input in [4096] {
                    for d_output in [4096] {
                        let config = LinearConfig::new(d_input, d_output)
                            .with_layout(layout)
                            .with_bias(bias);
                        let name = format!("{layout:?}-{bias:?}");
                        let bench = LinearBench::<B> {
                            shape: [2, 1, b, d_input].into(),
                            linear: config.init(device),
                            device: device.clone(),
                            name,
                        };
                        let result = run_benchmark(bench);
                        results.push(result);
                    }
                }
            }
        }
    }
    results
}

fn main() {
    burnbench::bench_on_backend!();
}
