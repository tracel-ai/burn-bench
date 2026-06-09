use std::io::{Read, Write};

use burn::tensor::{
    Bytes, DType, Device, Distribution, Shape, Tensor, TensorData,
    quantization::{
        BlockSize, QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue,
    },
};
use burnbench::{Benchmark, BenchmarkResult, run_benchmark};
use derive_new::new;

#[derive(new)]
struct ToDataBenchmark<const D: usize> {
    shape: Shape,
    device: Device,
}

struct TensorDesc {
    shape: Shape,
    quant: Option<QuantScheme>,
}

impl<const D: usize> Benchmark for ToDataBenchmark<D> {
    type Input = Tensor<D>;
    type Output = TensorData;

    fn name(&self) -> String {
        format!("to_data-{:?}", self.device.settings().float_dtype).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.to_vec()]
    }

    fn execute(&self, args: Self::Input) -> Self::Output {
        args.to_data()
    }

    fn prepare(&self) -> Self::Input {
        Tensor::random(self.shape.clone(), Distribution::Default, &self.device)
    }

    fn sync(&self) {
        self.device.sync().unwrap();
    }
}

struct FromFileBenchmark<const D: usize> {
    path: String,
    desc: TensorDesc,
    dtype: DType,
    len: usize,
    device: Device,
    lazy: bool,
}

impl<const D: usize> FromFileBenchmark<D> {
    pub fn new(shape: Shape, device: Device, lazy: bool, quant: Option<QuantScheme>) -> Self {
        let tensor = Tensor::<3>::random(shape.clone(), Distribution::Default, &device);
        let tensor = match quant {
            Some(scheme) => tensor.quantize_dynamic(&scheme),
            None => tensor,
        };
        let data = tensor.into_data();
        let len = data.bytes.len();
        let dtype = data.dtype;

        let path = format!("/tmp/{:?}-{:?}", device, shape);
        std::fs::remove_file(&path).ok();

        let mut file = std::fs::File::create_new(&path).unwrap();
        file.write_all(&data.bytes).unwrap();

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

impl<const D: usize> Benchmark for FromFileBenchmark<D> {
    type Input = ();
    type Output = Tensor<D>;

    fn name(&self) -> String {
        format!(
            "load-from-file-{:?}{}{}",
            self.device.settings().float_dtype,
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
        vec![self.desc.shape.to_vec()]
    }

    fn execute(&self, _input: Self::Input) -> Self::Output {
        if self.lazy {
            let bytes = Bytes::from_file(&self.path, self.len as u64, 0);
            let data = TensorData::from_bytes(bytes, self.desc.shape.clone(), self.dtype);
            Tensor::<D>::from_data(data, &self.device)
        } else {
            let mut file = std::fs::File::open(&self.path).unwrap();
            let mut buf = vec![0; self.len];
            file.read_exact(&mut buf).unwrap();
            let data = TensorData::from_bytes_vec(buf, self.desc.shape.clone(), self.dtype);
            Tensor::<D>::from_data(data, &self.device)
        }
    }

    fn prepare(&self) -> Self::Input {}

    fn sync(&self) {
        self.device.sync().unwrap();
    }
}

struct FromMemoryBenchmark<const D: usize> {
    data: TensorData,
    staging: bool,
    device: Device,
}

impl<const D: usize> FromMemoryBenchmark<D> {
    pub fn new(shape: Shape, device: Device, staging: bool) -> Self {
        let tensor = Tensor::<3>::random(shape.clone(), Distribution::Default, &device);
        let data = tensor.into_data();

        Self {
            data,
            device,
            staging,
        }
    }
}

impl<const D: usize> Benchmark for FromMemoryBenchmark<D> {
    type Input = TensorData;
    type Output = Tensor<D>;

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
        vec![self.data.shape.to_vec()]
    }

    fn execute(&self, data: Self::Input) -> Self::Output {
        Tensor::<D>::from_data(data, &self.device)
    }

    fn prepare(&self) -> Self::Input {
        let mut data = [self.data.clone()];
        if self.staging {
            self.device.staging(data.iter_mut());
        }
        let [data] = data;
        data
    }

    fn sync(&self) {
        self.device.sync().unwrap();
    }

    fn prepare_cloned(&self) -> bool {
        false
    }
}

#[allow(dead_code)]
fn bench(device: &Device) -> Vec<BenchmarkResult> {
    const D: usize = 3;
    let shape: Shape = [32, 512, 2048].into();
    let mut results = Vec::new();

    results.push(run_benchmark(ToDataBenchmark::<D>::new(
        shape.clone(),
        device.clone(),
    )));

    for staging in [true, false] {
        results.push(run_benchmark(FromMemoryBenchmark::<D>::new(
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
                store: QuantStore::PackedU32(0),
                level: QuantLevel::Block(BlockSize::new([32])),
                mode: QuantMode::Symmetric,
            }),
        ] {
            results.push(run_benchmark(FromFileBenchmark::<D>::new(
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
    let device = backend_comparison::select_device();
    let results = bench(&device);
    backend_comparison::save(results, &device);
}
