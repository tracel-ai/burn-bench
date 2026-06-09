use burn::tensor::quantization::{
    BlockSize, QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue,
};
use burn::tensor::{Device, Distribution, Shape, Tensor};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};
use core::f64::consts::SQRT_2;

#[derive(Debug)]
enum GeluKind {
    Reference,
    WithReferenceErf,
    WithCustomErf,
}

/// Benchmark how well a backend executes a custom activation function with a lot of basic tensor
/// operations.
struct CustomGeluBenchmark<const D: usize> {
    shape: Shape,
    device: Device,
    kind: GeluKind,
    mode: Mode,
}

#[derive(Clone, Copy)]
enum Mode {
    Autodiff,
    Inference(Option<QuantScheme>),
}

impl core::fmt::Debug for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Mode::Autodiff => f.write_str("autodiff"),
            Mode::Inference(scheme) => match scheme {
                Some(_) => f.write_str("inference-q4"),
                None => f.write_str("inference"),
            },
        }
    }
}

impl<const D: usize> CustomGeluBenchmark<D> {
    fn execute_autodiff(&self, tensor: Tensor<D>) -> Tensor<D> {
        // The input tensor is created on an autodiff-enabled device (see `bench`).
        let tensor = tensor.require_grad();
        let output = match self.kind {
            GeluKind::Reference => burn::tensor::activation::gelu(tensor.clone()),
            GeluKind::WithReferenceErf => gelu_custom(tensor.clone(), Tensor::erf),
            GeluKind::WithCustomErf => gelu_custom(tensor.clone(), erf_custom),
        };
        let mut gradients = output.sum().backward();
        tensor.grad_remove(&mut gradients).unwrap()
    }
}

impl<const D: usize> Benchmark for CustomGeluBenchmark<D> {
    type Input = Tensor<D>;
    type Output = Tensor<D>;

    fn name(&self) -> String {
        format!(
            "gelu-{:?}-{:?}-{:?}",
            self.kind,
            self.mode,
            self.device.settings().float_dtype
        )
        .to_lowercase()
    }

    fn options(&self) -> Option<String> {
        Some(format!("{:?}", self.kind))
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }

    fn execute(&self, tensor: Self::Input) -> Self::Output {
        match self.mode {
            Mode::Autodiff => self.execute_autodiff(tensor),
            Mode::Inference(_scheme) => match self.kind {
                GeluKind::Reference => burn::tensor::activation::gelu(tensor),
                GeluKind::WithReferenceErf => gelu_custom(tensor, Tensor::erf),
                GeluKind::WithCustomErf => gelu_custom(tensor, erf_custom),
            },
        }
    }

    fn prepare(&self) -> Self::Input {
        let input = Tensor::random(self.shape.clone(), Distribution::Default, &self.device);

        match &self.mode {
            Mode::Autodiff => input,
            Mode::Inference(scheme) => match scheme {
                Some(scheme) => input.quantize_dynamic(scheme),
                None => input,
            },
        }
    }

    fn sync(&self) {
        self.device.sync().unwrap();
    }

    fn num_samples(&self) -> usize {
        10
    }
}

fn gelu_custom<const D: usize, Erf>(x: Tensor<D>, erf: Erf) -> Tensor<D>
where
    Erf: Fn(Tensor<D>) -> Tensor<D>,
{
    let x = x.clone() * (erf(x / SQRT_2) + 1);
    x / 2
}

fn erf_custom<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    let x1 = -erf_positive(-x.clone());
    let x2 = erf_positive(x.clone());
    let mask = x.greater_elem(0);

    x1.mask_where(mask, x2)
}

/// An approximation of the error function: https://en.wikipedia.org/wiki/Error_function#Numerical_approximations
///
/// > (maximum error: 1.5×10−7)
/// > All of these approximations are valid for x ≥ 0. To use these approximations for negative x, use the fact that erf x is an odd function, so erf x = −erf(−x).
fn erf_positive<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    let p = 0.3275911;
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;

    let x1 = x.clone().abs() * p + 1;
    let t = x1.recip();
    let tmp = (((((t.clone() * a5) + a4) * t.clone()) + a3) * t.clone() + a2) * t.clone() + a1;

    -(tmp * t * (-x.clone() * x).exp()) + 1.0
}

#[allow(dead_code)]
fn bench(device: &Device) -> Vec<BenchmarkResult> {
    const D: usize = 3;
    let shape: Shape = [32, 512, 2048].into();

    let mut benches = Vec::new();
    let mut run = |mode: Mode| {
        // Autodiff is a property of the device.
        let device = match mode {
            Mode::Autodiff => device.clone().autodiff(),
            Mode::Inference(_) => device.clone(),
        };
        let reference_gelu = CustomGeluBenchmark::<D> {
            shape: shape.clone(),
            device: device.clone(),
            kind: GeluKind::Reference,
            mode,
        };
        let reference_erf_gelu = CustomGeluBenchmark::<D> {
            shape: shape.clone(),
            device: device.clone(),
            kind: GeluKind::WithReferenceErf,
            mode,
        };
        let custom_erf_gelu = CustomGeluBenchmark::<D> {
            shape: shape.clone(),
            device: device.clone(),
            kind: GeluKind::WithCustomErf,
            mode,
        };

        benches.push(run_benchmark(reference_gelu));
        benches.push(run_benchmark(reference_erf_gelu));
        benches.push(run_benchmark(custom_erf_gelu));
    };

    run(Mode::Inference(None));
    run(Mode::Inference(Some(QuantScheme {
        value: QuantValue::Q4F,
        param: QuantParam::F16,
        store: QuantStore::PackedU32(0),
        level: QuantLevel::Block(BlockSize::new([32])),
        mode: QuantMode::Symmetric,
    })));
    run(Mode::Autodiff);

    benches
}

fn main() {
    let device = backend_comparison::select_device();
    let results = bench(&device);
    backend_comparison::save(results, &device);
}
