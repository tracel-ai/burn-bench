use burn::tensor::{
    Device, Distribution, Shape, Tensor, module::conv_transpose2d, ops::ConvTransposeOptions,
};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};

pub struct ConvTranspose2dBenchmark {
    input_shape: Shape,
    weight_shape: Shape,
    bias_shape: Shape,
    options: ConvTransposeOptions<2>,
    device: Device,
}

impl Benchmark for ConvTranspose2dBenchmark {
    type Input = (Tensor<4>, Tensor<4>, Tensor<1>);
    type Output = Tensor<4>;

    fn name(&self) -> String {
        format!("conv_transpose2d-{:?}", self.device.settings().float_dtype).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![
            self.input_shape.to_vec(),
            self.weight_shape.to_vec(),
            self.bias_shape.to_vec(),
        ]
    }

    fn execute(&self, (x, w, b): Self::Input) -> Self::Output {
        conv_transpose2d(x, w, Some(b), self.options.clone())
    }

    fn prepare(&self) -> Self::Input {
        (
            Tensor::random(
                self.input_shape.clone(),
                Distribution::Default,
                &self.device,
            ),
            Tensor::random(
                self.weight_shape.clone(),
                Distribution::Default,
                &self.device,
            ),
            Tensor::random(self.bias_shape.clone(), Distribution::Default, &self.device),
        )
    }

    fn sync(&self) {
        self.device.sync().unwrap();
    }
}

#[allow(dead_code)]
fn bench(device: &Device) -> Vec<BenchmarkResult> {
    // Shapes
    let batch_size = 16;
    let channels_in = 16;
    let channels_out = 16;
    let height_in = 64;
    let width_in = 64;
    let kernel_size_0 = 8;
    let kernel_size_1 = 8;

    // Options
    let strides = [1, 1];
    let padding = [0, 0];
    let padding_out = [0, 0];
    let dilations = [1, 1];
    let groups = 1;
    let options = ConvTransposeOptions::new(strides, padding, padding_out, dilations, groups);
    let benchmark = ConvTranspose2dBenchmark {
        input_shape: [batch_size, channels_in, height_in, width_in].into(),
        weight_shape: [
            channels_in,
            channels_out / groups,
            kernel_size_0,
            kernel_size_1,
        ]
        .into(),
        bias_shape: [channels_out].into(),
        options,
        device: device.clone(),
    };

    vec![run_benchmark(benchmark)]
}

fn main() {
    let device = backend_comparison::select_device();
    let results = bench(&device);
    backend_comparison::save(results, &device);
}
