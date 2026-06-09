use burn::{
    module::Quantizer,
    nn,
    prelude::*,
    tensor::quantization::{
        BlockSize, Calibration, QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore,
        QuantValue,
    },
};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};

struct LinearBench {
    name: String,
    linear: nn::Linear,
    signal_shape: Shape,
    device: Device,
}

impl LinearBench {
    fn inference(config: nn::LinearConfig, device: &Device, batch_sizes: [usize; 2]) -> Self {
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
        device: &Device,
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
        device: &Device,
    ) -> (nn::Linear, Shape, String) {
        let signal_shape = Shape::new([batch_sizes[0], batch_sizes[1], config.d_input]);
        let name = match config.bias {
            true => "linear-bias",
            false => "linear",
        };
        let name = format!("{name}_{:?}", device.settings().float_dtype);
        let linear = config.init(device);

        (linear, signal_shape, name)
    }
}

impl Benchmark for LinearBench {
    type Input = Tensor<3>;
    type Output = Tensor<3>;

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
        self.device.sync().unwrap();
    }
    fn shapes(&self) -> Vec<Vec<usize>> {
        let weights = self.linear.weight.shape();
        vec![self.signal_shape.to_vec(), weights.to_vec()]
    }
}

#[allow(dead_code)]
fn bench(device: &Device) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    for (d_input, d_output) in [(4096, 4096)] {
        for bias in [true, false] {
            for batch_sizes in [[1, 1], [32, 1], [1, 32]] {
                let inference = LinearBench::inference(
                    nn::LinearConfig::new(d_input, d_output).with_bias(bias),
                    device,
                    batch_sizes,
                );
                results.push(run_benchmark(inference));

                #[allow(clippy::single_element_loop)]
                for (scheme, tag) in [(
                    QuantScheme {
                        value: QuantValue::Q4F,
                        param: QuantParam::F16,
                        store: QuantStore::PackedU32(0),
                        level: QuantLevel::Block(BlockSize::new([32])),
                        mode: QuantMode::Symmetric,
                    },
                    "q4b32",
                )] {
                    let inference = LinearBench::q_inference(
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
    let device = backend_comparison::select_device();
    let results = bench(&device);
    backend_comparison::save(results, &device);
}
