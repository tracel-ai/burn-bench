use burn::tensor::{Device, Distribution, Shape, Tensor};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};
use derive_new::new;

#[derive(new)]
struct ElemwiseBenchmark {
    shape: Shape,
    device: Device,
}

impl Benchmark for ElemwiseBenchmark {
    type Input = Tensor<3>;
    type Output = Tensor<3>;

    fn name(&self) -> String {
        format!("elemwise-{:?}", self.device.settings().float_dtype).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }

    fn execute(&self, input: Self::Input) -> Self::Output {
        let [a, b, c] = self.shape.dims();
        let device = input.device();
        let zeroes = Tensor::<3>::zeros([a, b, c * 2], &device);
        let fives = zeroes + 5;
        let view = input.reshape([1, 1, c, 1]);
        let tmp = view.clone().neg();
        let tmp = Tensor::cat(vec![tmp, view], 3);
        let tmp = tmp.reshape([1, 1, c * 2]);

        tmp * fives
    }

    fn prepare(&self) -> Self::Input {
        let [_, _, c] = self.shape.dims();
        Tensor::random([1, 1, c], Distribution::Default, &self.device)
    }

    fn sync(&self) {
        self.device.sync().unwrap();
    }
}

#[allow(dead_code)]
fn bench(device: &Device) -> Vec<BenchmarkResult> {
    let shape: Shape = [32, 4096, 128].into();

    let benchmark = ElemwiseBenchmark::new(shape, device.clone());

    vec![run_benchmark(benchmark)]
}

fn main() {
    let device = backend_comparison::select_device();
    let results = bench(&device);
    backend_comparison::save(results, &device);
}
