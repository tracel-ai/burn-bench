use burn::tensor::Int;
use burn::tensor::{Device, Distribution, Shape, Tensor};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};

enum Instruction {
    ArgMin(usize),
    ArgMinFused(usize),
    SumDim(usize),
    SumDimFused(usize),
    Sum,
}

struct ReduceBenchmark {
    instruction: Instruction,
    shape: Shape,
    device: Device,
    tensor: Tensor<3>,
}

impl ReduceBenchmark {
    pub fn new(instruction: Instruction, device: Device) -> Self {
        let shape = Shape::new([32, 512, 4096]);
        let tensor = Tensor::random(shape.clone(), Distribution::Default, &device);
        Self {
            instruction,
            shape,
            device,
            tensor,
        }
    }
}

pub enum ReduceOutput<const D: usize> {
    Arg(Tensor<D, Int>),
    Dim(Tensor<D>),
    Full(Tensor<1>),
}

impl Benchmark for ReduceBenchmark {
    type Input = ();
    type Output = ReduceOutput<3>;

    fn prepare(&self) -> Self::Input {}

    fn execute(&self, _: Self::Input) -> Self::Output {
        match self.instruction {
            Instruction::ArgMin(axis) => ReduceOutput::Arg(self.tensor.clone().argmin(axis)),
            Instruction::SumDim(axis) => ReduceOutput::Dim(self.tensor.clone().sum_dim(axis)),
            Instruction::SumDimFused(axis) => {
                let tensor = self.tensor.clone() + 5;
                let tensor = tensor.sum_dim(axis);
                let tensor = tensor.tanh();
                let tensor = tensor * 3;
                ReduceOutput::Dim(tensor)
            }
            Instruction::ArgMinFused(axis) => {
                let tensor = self.tensor.clone() + 5;
                let tensor = tensor.log();
                let tensor = tensor.tanh();
                let tensor = tensor * 3;
                ReduceOutput::Arg(tensor.argmin(axis))
            }
            Instruction::Sum => ReduceOutput::Full(self.tensor.clone().sum()),
        }
    }

    fn name(&self) -> String {
        match self.instruction {
            Instruction::ArgMin(axis) => {
                format!(
                    "reduce-argmin-{axis}-{:?}",
                    self.device.settings().float_dtype
                )
            }
            Instruction::ArgMinFused(axis) => {
                format!(
                    "reduce-argmin-{axis}-fused-{:?}",
                    self.device.settings().float_dtype
                )
            }
            Instruction::SumDim(axis) => {
                format!("reduce-sum-{axis}-{:?}", self.device.settings().float_dtype)
            }
            Instruction::SumDimFused(axis) => {
                format!(
                    "reduce-sum-{axis}-fused-{:?}",
                    self.device.settings().float_dtype
                )
            }
            Instruction::Sum => {
                format!("reduce-sum-full-{:?}", self.device.settings().float_dtype)
            }
        }
        .to_lowercase()
    }

    fn sync(&self) {
        self.device.sync().unwrap();
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }
}

#[allow(dead_code)]
fn bench(device: &Device) -> Vec<BenchmarkResult> {
    let mut benchmarks = Vec::new();

    for axis in 0..3 {
        benchmarks.push(ReduceBenchmark::new(
            Instruction::ArgMin(axis),
            device.clone(),
        ));
        benchmarks.push(ReduceBenchmark::new(
            Instruction::ArgMinFused(axis),
            device.clone(),
        ));
        benchmarks.push(ReduceBenchmark::new(
            Instruction::SumDim(axis),
            device.clone(),
        ));
        benchmarks.push(ReduceBenchmark::new(
            Instruction::SumDimFused(axis),
            device.clone(),
        ));
    }

    benchmarks.push(ReduceBenchmark::new(Instruction::Sum, device.clone()));
    benchmarks.into_iter().map(run_benchmark).collect()
}

fn main() {
    let device = backend_comparison::select_device();
    let results = bench(&device);
    backend_comparison::save(results, &device);
}
