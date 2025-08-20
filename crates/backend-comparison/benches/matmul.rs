use burn::tensor::{Distribution, Element, Shape, Tensor, backend::Backend};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};
use derive_new::new;

#[derive(new)]
struct MatmulBenchmark<B: Backend, const D: usize> {
    b: usize,
    problem: Problem,
    device: B::Device,
    layouts: (Layout, Layout),
}

#[derive(Clone, Copy)]
enum Problem {
    General {
        m: usize,
        n: usize,
        k: usize,
        lhs: Layout,
        rhs: Layout,
    },
    MatVec {
        m: usize,
        k: usize,
    },
    VecMat {
        n: usize,
        k: usize,
    },
    Inner {
        k: usize,
    },
    Outer {
        m: usize,
        n: usize,
    },
}

#[derive(Clone, Copy, Debug)]
enum Layout {
    Row,
    Col,
}

impl Problem {
    fn name(&self) -> &str {
        match self {
            Problem::General { .. } => "general",
            Problem::MatVec { .. } => "mat@vec",
            Problem::VecMat { .. } => "vec@mat",
            Problem::Inner { .. } => "inner",
            Problem::Outer { .. } => "outer",
        }
    }
    fn shapes(self, b: usize) -> (Shape, Shape) {
        match self {
            Problem::General { m, n, k, .. } => ([b, m, k].into(), [b, k, n].into()),
            Problem::MatVec { m, k } => ([b, m, k].into(), [b, k, 1].into()),
            Problem::VecMat { n, k } => ([b, 1, k].into(), [b, k, n].into()),
            Problem::Inner { k } => ([b, 1, k].into(), [b, k, 1].into()),
            Problem::Outer { m, n } => ([b, m, 1].into(), [b, 1, n].into()),
        }
    }
}

impl<B: Backend, const D: usize> Benchmark for MatmulBenchmark<B, D> {
    type Input = (Tensor<B, D>, Tensor<B, D>);
    type Output = Tensor<B, D>;

    fn name(&self) -> String {
        format!(
            "matmul-{}-{:?}-{:?}",
            self.problem.name(),
            B::FloatElem::dtype(),
            self.layouts
        )
        .to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        let (shape_lhs, shape_rhs) = self.problem.shapes(self.b);

        if shape_lhs == shape_rhs {
            vec![shape_lhs.dims]
        } else {
            vec![shape_lhs.dims, shape_rhs.dims]
        }
    }

    fn execute(&self, (lhs, rhs): Self::Input) -> Self::Output {
        lhs.matmul(rhs)
    }

    fn prepare(&self) -> Self::Input {
        let (shape_lhs, shape_rhs) = self.problem.shapes(self.b);
        let lhs = match self.layouts.0 {
            Layout::Row => Tensor::random(shape_lhs, Distribution::Default, &self.device),
            Layout::Col => {
                Tensor::random(transpose(shape_lhs), Distribution::Default, &self.device)
                    .transpose()
            }
        };
        let rhs = match self.layouts.1 {
            Layout::Row => Tensor::random(shape_rhs, Distribution::Default, &self.device),
            Layout::Col => {
                Tensor::random(transpose(shape_rhs), Distribution::Default, &self.device)
                    .transpose()
            }
        };

        (lhs, rhs)
    }

    fn sync(&self) {
        B::sync(&self.device)
    }
}

fn transpose(shape: Shape) -> Shape {
    let mut output = shape.clone();
    let rank = shape.num_dims();
    output.dims[rank - 1] = shape.dims[rank - 2];
    output.dims[rank - 2] = shape.dims[rank - 1];
    output
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();
    for lhs in [Layout::Row] {
        for rhs in [Layout::Row, Layout::Col] {
            for (b, problem) in [
                // General benches
                (
                    16,
                    Problem::General {
                        m: 1,
                        n: 6144,
                        k: 6144,
                        lhs,
                        rhs,
                    },
                ),
                (
                    16,
                    Problem::General {
                        m: 1,
                        n: 5000,
                        k: 5000,
                        lhs,
                        rhs,
                    },
                ),
                (
                    16,
                    Problem::General {
                        m: 1,
                        n: 2048,
                        k: 8192,
                        lhs,
                        rhs,
                    },
                ),
                (
                    16,
                    Problem::General {
                        m: 1,
                        n: 8192,
                        k: 2048,
                        lhs,
                        rhs,
                    },
                ),
            ] {
                let bench = MatmulBenchmark::<B, 3>::new(b, problem, device.clone(), (lhs, rhs));
                let result = run_benchmark(bench);
                results.push(result);
            }
        }
    }
    results
}

fn main() {
    burnbench::bench_on_backend!();
}
