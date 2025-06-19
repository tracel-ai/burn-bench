use burn::tensor::{Distribution, Element, Shape, Tensor, backend::Backend};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};
use derive_new::new;

#[derive(new)]
struct MatmulBenchmark<B: Backend, const D: usize> {
    b: usize,
    problem: Problem,
    device: B::Device,
}

#[derive(Clone, Copy)]
enum Problem {
    General { m: usize, n: usize, k: usize },
    MatVec { m: usize, k: usize },
    VecMat { n: usize, k: usize },
    Inner { k: usize },
    Outer { m: usize, n: usize },
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
            Problem::General { m, n, k } => ([b, m, k].into(), [b, k, n].into()),
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
        format!("matmul-{}-{:?}", self.problem.name(), B::FloatElem::dtype()).to_lowercase()
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
        let lhs = Tensor::random(shape_lhs, Distribution::Default, &self.device);
        let rhs = Tensor::random(shape_rhs, Distribution::Default, &self.device);

        (lhs, rhs)
    }

    fn sync(&self) {
        B::sync(&self.device)
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) -> Vec<BenchmarkResult> {
    [
        // General benches
        // (
        //     1,
        //     Problem::General {
        //         m: 6144,
        //         n: 6144,
        //         k: 6144,
        //     },
        // ),
        // (
        //     2,
        //     Problem::General {
        //         m: 5000,
        //         n: 5000,
        //         k: 5000,
        //     },
        // ),
        // (
        //     4,
        //     Problem::General {
        //         m: 4096,
        //         n: 4096,
        //         k: 4096,
        //     },
        // ),
        (
            4,
            Problem::General {
                m: 2048,
                n: 2048,
                k: 2048,
            },
        ),
        (
            8,
            Problem::General {
                m: 1024,
                n: 1024,
                k: 1024,
            },
        ),
        (
            16,
            Problem::General {
                m: 512,
                n: 512,
                k: 512,
            },
        ),
        (
            32,
            Problem::General {
                m: 256,
                n: 256,
                k: 256,
            },
        ),
        // Mat@Vec benches
        (1, Problem::MatVec { m: 8192, k: 8192 }),
        (2, Problem::MatVec { m: 8192, k: 8192 }),
        // Vec@Mat benches
        (1, Problem::VecMat { n: 8192, k: 8192 }),
        (2, Problem::VecMat { n: 8192, k: 8192 }),
        // Inner benches
        (1, Problem::Inner { k: 8192 }),
        // Outer benches
        (
            1,
            Problem::Outer {
                m: 8192 * 2,
                n: 8192 * 2,
            },
        ),
        (4, Problem::Outer { m: 8192, n: 8192 }),
    ]
    .into_iter()
    .map(|(b, problem)| MatmulBenchmark::<B, 3>::new(b, problem, device.clone()))
    .map(run_benchmark)
    .collect()
}

fn main() {
    burnbench::bench_on_backend!();
}
