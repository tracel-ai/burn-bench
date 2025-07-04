use burn::tensor::Int;
use burn::tensor::{Distribution, Element, Shape, Tensor, backend::Backend};
use burnbench;
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};

enum Instruction {
    ArgMin(usize),
    ArgMinFused(usize),
    SumDim(usize),
    SumDimFused(usize),
    Sum,
}

struct ReduceBenchmark<B: Backend> {
    instruction: Instruction,
    shape: Shape,
    device: B::Device,
    tensor: Tensor<B, 3>,
}

impl<B: Backend> ReduceBenchmark<B> {
    pub fn new(instruction: Instruction, device: B::Device) -> Self {
        let shape = Shape::new([2048, 256, 64]);
        let tensor = Tensor::random(shape.clone(), Distribution::Default, &device);
        Self {
            instruction,
            shape,
            device,
            tensor,
        }
    }
}

pub enum ReduceOutput<B: Backend, const D: usize> {
    Arg(Tensor<B, D, Int>),
    Dim(Tensor<B, D>),
    Full(Tensor<B, 1>),
}

impl<B: Backend> Benchmark for ReduceBenchmark<B> {
    type Input = ();
    type Output = ReduceOutput<B, 3>;

    fn prepare(&self) -> Self::Input {}

    fn execute(&self, _: Self::Input) -> Self::Output {
        match self.instruction {
            Instruction::ArgMin(axis) => ReduceOutput::Arg(self.tensor.clone().argmin(axis)),
            Instruction::SumDim(axis) => ReduceOutput::Dim(self.tensor.clone().sum_dim(axis)),
            Instruction::SumDimFused(axis) => {
                let tensor = self.tensor.clone() + 5;
                let tensor = tensor.log();
                let tensor = tensor.tanh();
                let tensor = tensor * 3;
                ReduceOutput::Dim(tensor.sum_dim(axis))
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
                format!("reduce-argmin-{axis}-{:?}", B::FloatElem::dtype())
            }
            Instruction::ArgMinFused(axis) => {
                format!("reduce-argmin-{axis}-fused-{:?}", B::FloatElem::dtype())
            }
            Instruction::SumDim(axis) => format!("reduce-sum-{axis}-{:?}", B::FloatElem::dtype()),
            Instruction::SumDimFused(axis) => {
                format!("reduce-sum-{axis}-fused-{:?}", B::FloatElem::dtype())
            }
            Instruction::Sum => format!("reduce-sum-full-{:?}", B::FloatElem::dtype()),
        }
        .to_lowercase()
    }

    fn sync(&self) {
        B::sync(&self.device)
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.dims.clone()]
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) -> Vec<BenchmarkResult> {
    let mut benchmarks = Vec::new();

    for axis in 0..3 {
        benchmarks.push(ReduceBenchmark::<B>::new(
            Instruction::ArgMin(axis),
            device.clone(),
        ));
        benchmarks.push(ReduceBenchmark::<B>::new(
            Instruction::ArgMinFused(axis),
            device.clone(),
        ));

        benchmarks.push(ReduceBenchmark::<B>::new(
            Instruction::SumDim(axis),
            device.clone(),
        ));
        benchmarks.push(ReduceBenchmark::<B>::new(
            Instruction::SumDimFused(axis),
            device.clone(),
        ));
    }

    benchmarks.push(ReduceBenchmark::<B>::new(Instruction::Sum, device.clone()));
    benchmarks.into_iter().map(run_benchmark).collect()
}

fn main() {
    burnbench::bench_on_backend!()
}
