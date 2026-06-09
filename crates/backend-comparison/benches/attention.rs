use std::fmt::Display;

use burn::tensor::{
    Bool, Device, Distribution, Shape, Tensor,
    module::{attention, attention_fallback},
    ops::AttentionModuleOptions,
};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};

pub struct AttentionBenchmark {
    problem: AttentionProblem,
    kind: AttentionKind,
    device: Device,
}

#[derive(Clone)]
struct AttentionProblem {
    batch_size: usize,
    num_heads: usize,
    seq_q: usize,
    head_dim: usize,
    seq_kv: usize,
    val_dim: usize,
    mask: bool,
    options: AttentionModuleOptions,
}

impl AttentionProblem {
    fn query_shape(&self) -> Shape {
        Shape::new([self.batch_size, self.num_heads, self.seq_q, self.head_dim])
    }

    fn key_shape(&self) -> Shape {
        Shape::new([self.batch_size, self.num_heads, self.seq_kv, self.head_dim])
    }

    fn value_shape(&self) -> Shape {
        Shape::new([self.batch_size, self.num_heads, self.seq_kv, self.val_dim])
    }

    fn mask_shape(&self) -> Shape {
        Shape::new([self.batch_size, self.num_heads, self.seq_q, self.seq_kv])
    }
}

enum AttentionKind {
    Flash,
    Fallback,
}

impl Display for AttentionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AttentionKind::Flash => f.write_str("flash"),
            AttentionKind::Fallback => f.write_str("fallback"),
        }
    }
}

#[derive(Clone)]
pub struct AttentionInput {
    query: Tensor<4>,
    key: Tensor<4>,
    value: Tensor<4>,
    mask: Option<Tensor<4, Bool>>,
}

impl AttentionInput {
    fn new(problem: &AttentionProblem, device: &Device) -> Self {
        let query = Tensor::random(problem.query_shape(), Distribution::Default, device);
        let key = Tensor::random(problem.key_shape(), Distribution::Default, device);
        let value = Tensor::random(problem.value_shape(), Distribution::Default, device);
        let mask = problem.mask.then(|| {
            Tensor::<4>::random(problem.mask_shape(), Distribution::Default, device).bool()
        });
        AttentionInput {
            query,
            key,
            value,
            mask,
        }
    }
}

impl Benchmark for AttentionBenchmark {
    type Input = AttentionInput;
    type Output = Tensor<4>;

    fn name(&self) -> String {
        format!(
            "attention_{}-{:?}",
            self.kind,
            self.device.settings().float_dtype
        )
        .to_lowercase()
    }

    fn execute(&self, input: Self::Input) -> Self::Output {
        match self.kind {
            AttentionKind::Flash => attention(
                input.query,
                input.key,
                input.value,
                input.mask,
                None,
                self.problem.options,
            ),
            AttentionKind::Fallback => attention_fallback(
                input.query,
                input.key,
                input.value,
                input.mask,
                None,
                self.problem.options,
            ),
        }
    }

    fn prepare(&self) -> Self::Input {
        AttentionInput::new(&self.problem, &self.device)
    }

    fn sync(&self) {
        self.device.sync().unwrap();
    }
}

#[allow(dead_code)]
fn bench(device: &Device) -> Vec<BenchmarkResult> {
    let small_problem = AttentionProblem {
        batch_size: 1,
        num_heads: 4,
        seq_q: 2048,
        head_dim: 128,
        seq_kv: 2048,
        val_dim: 128,
        mask: false,
        options: AttentionModuleOptions {
            scale: None,
            softcap: None,
            is_causal: false,
        },
    };

    let benchmark_flash = AttentionBenchmark {
        device: device.clone(),
        problem: small_problem.clone(),
        kind: AttentionKind::Flash,
    };
    let benchmark_fallback = AttentionBenchmark {
        device: device.clone(),
        problem: small_problem,
        kind: AttentionKind::Fallback,
    };

    vec![
        run_benchmark(benchmark_flash),
        run_benchmark(benchmark_fallback),
    ]
}

fn main() {
    let device = backend_comparison::select_device();
    let results = bench(&device);
    backend_comparison::save(results, &device);
}
