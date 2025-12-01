use burn::{
    module::Quantizer,
    prelude::*,
    tensor::Element,
    tensor::quantization::{
        BlockSize, Calibration, QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore,
        QuantValue,
    },
};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};

struct LinearBench<B: Backend> {
    name: String,
    linear: nn::Linear<B>,
    signal_shape: Shape,
    device: Device<B>,
}

impl<B: Backend> LinearBench<B> {
    fn inference(config: nn::LinearConfig, device: &Device<B>, batch_sizes: [usize; 2]) -> Self {
        let (linear, signal_shape, name) = Self::init(config, batch_sizes, device);

        Self {
            name,
            linear,
            signal_shape,
            device: device.clone(),
        }
    }

    fn q_inference(
        config: nn::LinearConfig,
        device: &Device<B>,
        scheme: QuantScheme,
        scheme_tag: &str,
        batch_sizes: [usize; 2],
    ) -> Self {
        let (linear, signal_shape, name) = Self::init(config, batch_sizes, device);
        let calibration = Calibration::MinMax;
        let mut quantizer = Quantizer {
            calibration,
            scheme,
        };
        let linear = linear.quantize_weights(&mut quantizer);

        Self {
            name: format!("q_{name}_{scheme_tag}"),
            linear,
            signal_shape,
            device: device.clone(),
        }
    }

    fn init(
        config: nn::LinearConfig,
        batch_sizes: [usize; 2],
        device: &Device<B>,
    ) -> (nn::Linear<B>, Shape, String) {
        let signal_shape = Shape::new([batch_sizes[0], batch_sizes[1], config.d_input]);
        let name = match config.bias {
            true => "linear-bias",
            false => "linear",
        };
        let name = format!("{name}_{:?}", B::FloatElem::dtype());
        let linear = config.init(device);

        (linear, signal_shape, name)
    }
}

impl<B: Backend> Benchmark for LinearBench<B> {
    type Input = Tensor<B, 3>;
    type Output = Tensor<B, 3>;

    fn prepare(&self) -> Self::Input {
        Tensor::random(
            self.signal_shape.clone(),
            burn::tensor::Distribution::Default,
            &self.device,
        )
    }

    fn execute(&self, input: Self::Input) -> Self::Output {
        self.linear.forward(input)
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn sync(&self) {
        B::sync(&self.device).unwrap();
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        let weights = self.linear.weight.shape();
        vec![self.signal_shape.to_vec(), weights.to_vec()]
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    for (d_input, d_output) in [(4096, 4096)] {
        for bias in [true, false] {
            for batch_sizes in [[1, 1], [32, 1], [1, 32]] {
                let inference = LinearBench::<B>::inference(
                    nn::LinearConfig::new(d_input, d_output).with_bias(bias),
                    device,
                    batch_sizes,
                );
                results.push(run_benchmark(inference));

                for (scheme, tag) in [(
                    QuantScheme {
                        value: QuantValue::Q4F,
                        param: QuantParam::F16,
                        store: QuantStore::U32,
                        level: QuantLevel::Block(BlockSize::new([32])),
                        mode: QuantMode::Symmetric,
                    },
                    "q4b32",
                )] {
                    let inference = LinearBench::<B>::q_inference(
                        nn::LinearConfig::new(d_input, d_output).with_bias(bias),
                        device,
                        scheme,
                        tag,
                        batch_sizes,
                    );
                    results.push(run_benchmark(inference));
                }
            }
        }
    }

    results
}

fn main() {
    burnbench::bench_on_backend!();
}
