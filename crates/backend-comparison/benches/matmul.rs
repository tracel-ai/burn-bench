use burn::tensor::{Device, Distribution, FloatDType, Shape, Tensor};
use burnbench::{Benchmark, BenchmarkResult, Compute, DType, Limit, run_benchmark};
use derive_new::new;

/// Maps a burn float dtype to the burn-free dtype used in benchmark declarations.
fn to_dtype(dtype: FloatDType) -> DType {
    match dtype {
        FloatDType::F64 => DType::F64,
        FloatDType::F32 => DType::F32,
        FloatDType::Flex32 => DType::Flex32,
        FloatDType::F16 => DType::F16,
        FloatDType::BF16 => DType::BF16,
    }
}

#[derive(new)]
struct MatmulBenchmark<const D: usize> {
    problem: Problem,
    device: Device,
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

    /// Effective `(batch, m, n, k)` of the matmul, with batch broadcast applied.
    fn matmul_dims(self) -> (usize, usize, usize, usize) {
        match self {
            Problem::General { b, m, n, k } => (b, m, n, k),
            Problem::MatVec { b, m, k } => (b, m, 1, k),
            Problem::VecMat { b_lhs, b_rhs, n, k } => (b_lhs.max(b_rhs), 1, n, k),
            Problem::Inner { b, k } => (b, 1, 1, k),
            Problem::Outer { b, m, n } => (b, m, n, 1),
        }
    }

    /// Best-case FLOPs: one multiply and one add per accumulated element.
    fn flops(self) -> u64 {
        let (b, m, n, k) = self.matmul_dims();
        2 * b as u64 * m as u64 * n as u64 * k as u64
    }

    /// Best-case element traffic: both inputs read once plus the output written
    /// once.
    fn elements(self) -> u64 {
        let (shape_lhs, shape_rhs) = self.shapes();
        let prod = |shape: &Shape| shape.to_vec().iter().map(|&d| d as u64).product::<u64>();
        let (b, m, n, _k) = self.matmul_dims();
        prod(&shape_lhs) + prod(&shape_rhs) + b as u64 * m as u64 * n as u64
    }
}

impl<const D: usize> Benchmark for MatmulBenchmark<D> {
    type Input = (Tensor<D>, Tensor<D>);
    type Output = Tensor<D>;

    fn name(&self) -> String {
        format!(
            "matmul-{}-{:?}",
            self.problem.name(),
            self.device.settings().float_dtype
        )
        .to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        let (shape_lhs, shape_rhs) = self.problem.shapes();

        if shape_lhs == shape_rhs {
            vec![shape_lhs.to_vec()]
        } else {
            vec![shape_lhs.to_vec(), shape_rhs.to_vec()]
        }
    }

    fn limits(&self) -> Limit {
        let float_dtype = self.device.settings().float_dtype;
        let elem_size = burn::tensor::DType::from(float_dtype).size() as u64;
        // A dense matmul is tensor-core work: declare it against the canonical
        // 16x16x16 MMA tile. Calibration measures the real tensor-core peak for
        // dtypes whose hardware supports that tile (f16/bf16) and reports N/A for
        // the tensor-core column otherwise (e.g. f32); the arithmetic column is
        // always scored against the measured scalar peak for the dtype.
        Limit {
            memory: Some(self.problem.elements() * elem_size),
            compute: vec![Compute::TensorCore {
                dtype: to_dtype(float_dtype),
                mnk: [16, 16, 16],
                count: self.problem.flops(),
            }],
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
        self.device.sync().unwrap();
    }
}

#[allow(dead_code)]
fn bench(device: &Device) -> Vec<BenchmarkResult> {
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
    .map(|problem| MatmulBenchmark::<3>::new(problem, device.clone()))
    .map(run_benchmark)
    .collect()
}

fn main() {
    let device = backend_comparison::select_device();
    let results = bench(&device);
    backend_comparison::save(results, &device);
}
