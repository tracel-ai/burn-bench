use burn::tensor::{Distribution, Element, Shape, Tensor, backend::Backend};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};
use derive_new::new;

#[derive(new)]
struct MatmulBenchmark<B: Backend, const D: usize> {
    problem: Problem,
    device: B::Device,
}

#[derive(Clone, Copy)]
enum Problem {
    General {
        b: usize,
        m: usize,
        n: usize,
        k: usize,
    },
    MatVec {
        b: usize,
        m: usize,
        k: usize,
    },
    VecMat {
        b_lhs: usize,
        b_rhs: usize,
        n: usize,
        k: usize,
    },
    Inner {
        b: usize,
        k: usize,
    },
    Outer {
        b: usize,
        m: usize,
        n: usize,
    },
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
    fn shapes(self) -> (Shape, Shape) {
        match self {
            Problem::General { b, m, n, k } => ([b, m, k].into(), [b, k, n].into()),
            Problem::MatVec { b, m, k } => ([b, m, k].into(), [b, k, 1].into()),
            Problem::VecMat { b_lhs, b_rhs, n, k } => ([b_lhs, 1, k].into(), [b_rhs, k, n].into()),
            Problem::Inner { b, k } => ([b, 1, k].into(), [b, k, 1].into()),
            Problem::Outer { b, m, n } => ([b, m, 1].into(), [b, 1, n].into()),
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
        let (shape_lhs, shape_rhs) = self.problem.shapes();

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
        let (shape_lhs, shape_rhs) = self.problem.shapes();
        let lhs = Tensor::random(shape_lhs, Distribution::Default, &self.device);
        let rhs = Tensor::random(shape_rhs, Distribution::Default, &self.device);

        (lhs, rhs)
    }

    fn sync(&self) {
        B::sync(&self.device).unwrap();
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) -> Vec<BenchmarkResult> {
    [
        // General benches
        Problem::General {
            b: 1,
            m: 6144,
            n: 6144,
            k: 6144,
        },
        Problem::General {
            b: 2,
            m: 5000,
            n: 5000,
            k: 5000,
        },
        Problem::General {
            b: 4,
            m: 4096,
            n: 4096,
            k: 4096,
        },
        Problem::General {
            b: 4,
            m: 2048,
            n: 2048,
            k: 2048,
        },
        Problem::General {
            b: 8,
            m: 1024,
            n: 1024,
            k: 1024,
        },
        Problem::General {
            b: 16,
            m: 512,
            n: 512,
            k: 512,
        },
        Problem::General {
            b: 32,
            m: 256,
            n: 256,
            k: 256,
        },
        // Mat@Vec benches
        Problem::MatVec {
            b: 1,
            m: 8192,
            k: 8192,
        },
        Problem::MatVec {
            b: 2,
            m: 8192,
            k: 8192,
        },
        // Vec@Mat benches
        Problem::VecMat {
            b_lhs: 1,
            b_rhs: 1,
            n: 8192,
            k: 8192,
        },
        Problem::VecMat {
            b_lhs: 2,
            b_rhs: 2,
            n: 8192,
            k: 8192,
        },
        // Should be treated as a general matmul
        Problem::General {
            b: 1,
            m: 4096,
            n: 4096,
            k: 4096,
        },
        Problem::VecMat {
            b_lhs: 4096,
            b_rhs: 1,
            n: 4096,
            k: 4096,
        },
        // Inner benches
        Problem::Inner { b: 1, k: 8192 },
        // Outer benches
        Problem::Outer {
            b: 1,
            m: 8192 * 2,
            n: 8192 * 2,
        },
        Problem::Outer {
            b: 4,
            m: 8192,
            n: 8192,
        },
    ]
    .into_iter()
    .map(|problem| MatmulBenchmark::<B, 3>::new(problem, device.clone()))
    .map(run_benchmark)
    .collect()
}

fn main() {
    burnbench::bench_on_backend!();
}
