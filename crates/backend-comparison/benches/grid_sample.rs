use burn::tensor::{Distribution, Element, Shape, Tensor, backend::Backend};
use burnbench;
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};

struct GridSampleBenchmark<B: Backend> {
    n_batch: usize,
    channels: usize,
    width_in: usize,
    height_in: usize,
    width_out: usize,
    height_out: usize,
    device: B::Device,
}

impl<B: Backend> GridSampleBenchmark<B> {
    pub fn new(
        n_batch: usize,
        channels: usize,
        width_in: usize,
        height_in: usize,
        width_out: usize,
        height_out: usize,
        device: B::Device,
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

impl<B: Backend> Benchmark for GridSampleBenchmark<B> {
    type Input = (Tensor<B, 4>, Tensor<B, 4>);
    type Output = Tensor<B, 4>;

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
            B::FloatElem::dtype()
        )
    }

    fn sync(&self) {
        B::sync(&self.device).unwrap();
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
fn bench<B: Backend>(device: &B::Device) -> Vec<BenchmarkResult> {
    let benchmarks = vec![
        GridSampleBenchmark::<B>::new(1, 1, 64, 64, 4, 4, device.clone()),
        GridSampleBenchmark::<B>::new(1, 1, 4, 4, 64, 64, device.clone()),
    ];

    benchmarks.into_iter().map(run_benchmark).collect()
}

fn main() {
    burnbench::bench_on_backend!()
}
