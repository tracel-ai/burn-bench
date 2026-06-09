use burn::tensor::{
    Device, Distribution, Shape, Tensor, module::conv_transpose3d, ops::ConvTransposeOptions,
};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};

pub struct ConvTranspose3dBenchmark {
    input_shape: Shape,
    weight_shape: Shape,
    bias_shape: Shape,
    options: ConvTransposeOptions<3>,
    device: Device,
}

impl Benchmark for ConvTranspose3dBenchmark {
    type Input = (Tensor<5>, Tensor<5>, Tensor<1>);
    type Output = Tensor<5>;

    fn name(&self) -> String {
        format!("conv_transpose3d-{:?}", self.device.settings().float_dtype).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![
            self.input_shape.to_vec(),
            self.weight_shape.to_vec(),
            self.bias_shape.to_vec(),
        ]
    }

    fn execute(&self, (x, w, b): Self::Input) -> Self::Output {
        conv_transpose3d(x, w, Some(b), self.options.clone())
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
    let depth_in = 4;
    let height_in = 16;
    let width_in = 16;
    let kernel_size_0 = 8;
    let kernel_size_1 = 8;
    let kernel_size_2 = 8;

    // Options
    let strides = [1, 1, 1];
    let padding = [0, 0, 0];
    let padding_out = [0, 0, 0];
    let dilations = [1, 1, 1];
    let groups = 1;
    let options = ConvTransposeOptions::new(strides, padding, padding_out, dilations, groups);
    let benchmark = ConvTranspose3dBenchmark {
        input_shape: [batch_size, channels_in, depth_in, height_in, width_in].into(),
        weight_shape: [
            channels_in,
            channels_out / groups,
            kernel_size_0,
            kernel_size_1,
            kernel_size_2,
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
