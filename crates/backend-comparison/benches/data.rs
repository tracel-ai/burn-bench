use std::io::{Read, Write};

use burn::tensor::{
    Bytes, DType, Distribution, Element, Shape, Tensor, TensorData,
    backend::Backend,
    quantization::{
        BlockSize, QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue,
    },
};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};
use derive_new::new;

#[derive(new)]
struct ToDataBenchmark<B: Backend, const D: usize> {
    shape: Shape,
    device: B::Device,
}

struct TensorDesc {
    shape: Shape,
    quant: Option<QuantScheme>,
}

impl<B: Backend, const D: usize> Benchmark for ToDataBenchmark<B, D> {
    type Input = Tensor<B, D>;
    type Output = TensorData;

    fn name(&self) -> String {
        format!("to_data-{:?}", B::FloatElem::dtype()).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.dims.clone()]
    }

    fn execute(&self, args: Self::Input) -> Self::Output {
        args.to_data()
    }

    fn prepare(&self) -> Self::Input {
        Tensor::random(self.shape.clone(), Distribution::Default, &self.device)
    }

    fn sync(&self) {
        B::sync(&self.device).unwrap();
    }
}

struct FromFileBenchmark<B: Backend, const D: usize> {
    path: String,
    desc: TensorDesc,
    dtype: DType,
    len: usize,
    device: B::Device,
    lazy: bool,
}

impl<B: Backend, const D: usize> FromFileBenchmark<B, D> {
    pub fn new(shape: Shape, device: B::Device, lazy: bool, quant: Option<QuantScheme>) -> Self {
        let tensor = Tensor::<B, 3>::random(shape.clone(), Distribution::Default, &device);
        let tensor = match quant {
            Some(scheme) => tensor.quantize_dynamic(&scheme),
            None => tensor,
        };
        let data = tensor.into_data();
        let len = data.bytes.len();
        let dtype = data.dtype;

        let path = format!("/tmp/{:?}-{:?}", B::name(&device), shape);
        std::fs::remove_file(&path).ok();

        let mut file = std::fs::File::create_new(&path).unwrap();
        file.write(&data.bytes).unwrap();

        Self {
            path,
            device,
            dtype,
            len,
            desc: TensorDesc { shape, quant },
            lazy,
        }
    }
}

impl<B: Backend, const D: usize> Benchmark for FromFileBenchmark<B, D> {
    type Input = ();
    type Output = Tensor<B, D>;

    fn name(&self) -> String {
        format!(
            "load-from-file-{:?}{}{}",
            B::FloatElem::dtype(),
            match self.lazy {
                true => "-lazy",
                false => "",
            },
            match self.desc.quant.is_some() {
                true => "-quant",
                false => "",
            }
        )
        .to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.desc.shape.dims.clone()]
    }

    fn execute(&self, _input: Self::Input) -> Self::Output {
        if self.lazy {
            let bytes = Bytes::from_file(&self.path, self.len as u64, 0);
            let data = TensorData::from_bytes(bytes, self.desc.shape.clone(), self.dtype);
            Tensor::<B, D>::from_data(data, &self.device)
        } else {
            let mut file = std::fs::File::open(&self.path).unwrap();
            let mut buf = vec![0; self.len];
            file.read_exact(&mut buf).unwrap();
            let data = TensorData::from_bytes_vec(buf, self.desc.shape.clone(), self.dtype);
            Tensor::<B, D>::from_data(data, &self.device)
        }
    }

    fn prepare(&self) -> Self::Input {
        ()
    }

    fn sync(&self) {
        B::sync(&self.device).unwrap();
    }
}

struct FromMemoryBenchmark<B: Backend, const D: usize> {
    data: TensorData,
    staging: bool,
    device: B::Device,
}

impl<B: Backend, const D: usize> FromMemoryBenchmark<B, D> {
    pub fn new(shape: Shape, device: B::Device, staging: bool) -> Self {
        let tensor = Tensor::<B, 3>::random(shape.clone(), Distribution::Default, &device);
        let data = tensor.into_data();

        Self {
            data,
            device,
            staging,
        }
    }
}

impl<B: Backend, const D: usize> Benchmark for FromMemoryBenchmark<B, D> {
    type Input = TensorData;
    type Output = Tensor<B, D>;

    fn name(&self) -> String {
        format!(
            "load-from-memory{}",
            match self.staging {
                true => "-staging",
                false => "",
            }
        )
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.data.shape.clone()]
    }

    fn execute(&self, data: Self::Input) -> Self::Output {
        Tensor::<B, D>::from_data(data, &self.device)
    }

    fn prepare(&self) -> Self::Input {
        let mut data = [self.data.clone()];
        if self.staging {
            B::staging(data.iter_mut(), &self.device);
        }
        let [data] = data;
        data
    }

    fn sync(&self) {
        B::sync(&self.device).unwrap();
    }

    fn prepare_cloned(&self) -> bool {
        false
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) -> Vec<BenchmarkResult> {
    const D: usize = 3;
    let shape: Shape = [32, 512, 2048].into();
    let mut results = Vec::new();

    results.push(run_benchmark(ToDataBenchmark::<B, D>::new(
        shape.clone(),
        device.clone(),
    )));

    for staging in [true, false] {
        results.push(run_benchmark(FromMemoryBenchmark::<B, D>::new(
            shape.clone(),
            device.clone(),
            staging,
        )));
    }

    for lazy in [true, false] {
        for quant in [
            None,
            Some(QuantScheme {
                value: QuantValue::Q4F,
                param: QuantParam::F16,
                store: QuantStore::U32,
                level: QuantLevel::Block(BlockSize::new([32])),
                mode: QuantMode::Symmetric,
            }),
        ] {
            results.push(run_benchmark(FromFileBenchmark::<B, D>::new(
                shape.clone(),
                device.clone(),
                lazy,
                quant,
            )));
        }
    }

    results
}

fn main() {
    burnbench::bench_on_backend!();
}
