use burn::tensor::{Device, Distribution, Shape, Tensor};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};

struct GridSampleBenchmark {
    n_batch: usize,
    channels: usize,
    width_in: usize,
    height_in: usize,
    width_out: usize,
    height_out: usize,
    device: Device,
}

impl GridSampleBenchmark {
    pub fn new(
        n_batch: usize,
        channels: usize,
        width_in: usize,
        height_in: usize,
        width_out: usize,
        height_out: usize,
        device: Device,
    ) -> Self {
        Self {
            n_batch,
            channels,
            width_in,
            height_in,
            width_out,
            height_out,
            device,
        }
    }
}

impl Benchmark for GridSampleBenchmark {
    type Input = (Tensor<4>, Tensor<4>);
    type Output = Tensor<4>;

    fn prepare(&self) -> Self::Input {
        let tensor = Tensor::random(
            Shape::new([self.n_batch, self.channels, self.width_in, self.height_in]),
            Distribution::Default,
            &self.device,
        );
        let grid = Tensor::random(
            Shape::new([self.n_batch, self.width_out, self.height_out, 2]),
            Distribution::Uniform(-1.0, 1.0),
            &self.device,
        );
        (tensor, grid)
    }

    fn execute(&self, (tensor, grid): Self::Input) -> Self::Output {
        tensor.clone().grid_sample_2d(
            grid.clone(),
            burn::tensor::ops::GridSampleOptions::new(burn::tensor::ops::InterpolateMode::Nearest),
        )
    }

    fn name(&self) -> String {
        format!(
            "grid-sample-b{}-c{}-in{}x{}-out{}x{}-{:?}",
            self.n_batch,
            self.channels,
            self.width_in,
            self.height_in,
            self.width_out,
            self.height_out,
            self.device.settings().float_dtype
        )
    }

    fn sync(&self) {
        self.device.sync().unwrap();
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![
            self.n_batch,
            self.channels,
            self.width_in,
            self.height_in,
            self.width_out,
            self.height_out,
        ]]
    }
}

#[allow(dead_code)]
fn bench(device: &Device) -> Vec<BenchmarkResult> {
    let benchmarks = vec![
        GridSampleBenchmark::new(1, 1, 64, 64, 4, 4, device.clone()),
        GridSampleBenchmark::new(1, 1, 4, 4, 64, 64, device.clone()),
    ];

    benchmarks.into_iter().map(run_benchmark).collect()
}

fn main() {
    let device = backend_comparison::select_device();
    let results = bench(&device);
    backend_comparison::save(results, &device);
}
