use burn::tensor::backend::Backend;
use burnbench;
use burnbench::BenchmarkResult;

#[cfg(all(
    feature = "test-remote",
    not(feature = "legacy-v16"),
    not(feature = "legacy-v17")
))]
mod remote_benchmarks {
    use super::*;

    use burnbench::{Benchmark, run_benchmark};
    use std::marker::PhantomData;

    use burn::backend::remote::{self, RemoteDevice};
    use burn::{
        backend::BackendIr,
        tensor::{Distribution, Shape, Tensor, backend::Backend},
    };
    use tokio::runtime::Runtime;

    struct LocalServer<B: BackendIr> {
        port: u16,
        runtime: Runtime,
        _phantom_data: PhantomData<B>,
    }

    impl<B: BackendIr> LocalServer<B> {
        pub fn new(port: u16) -> Self {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_io()
                .build()
                .unwrap();

            runtime.spawn(burn::server::start_async::<B>(Default::default(), port));

            Self {
                port,
                runtime,
                _phantom_data: Default::default(),
            }
        }

        pub fn get_device(&self) -> RemoteDevice {
            println!("ws://localhost:{}", self.port);
            remote::RemoteDevice::new(&format!("ws://localhost:{}", self.port))
        }
    }

    struct RemoteBenchmark<'a, B: BackendIr> {
        shape: Shape,
        device_a: &'a RemoteDevice,
        device_b: &'a RemoteDevice,
        _phantom_data: PhantomData<B>,
    }

    impl<'a, B: BackendIr> RemoteBenchmark<'a, B> {
        pub fn new(shape: Shape, device_a: &'a RemoteDevice, device_b: &'a RemoteDevice) -> Self {
            Self {
                shape,
                device_a,
                device_b,
                _phantom_data: Default::default(),
            }
        }
    }

    impl<'a, B: burn::backend::BackendIr> Benchmark for RemoteBenchmark<'a, B> {
        type Input = ();
        type Output = ();

        fn prepare(&self) -> Self::Input {}

        fn execute(&self, _: Self::Input) -> Self::Output {
            // Some random input
            let input = Tensor::<remote::RemoteBackend, 3>::random(
                self.shape.clone(),
                Distribution::Default,
                &self.device_a,
            );
            let numbers_expected: Vec<f32> = input.to_data().to_vec().unwrap();

            // Move tensor to device 2
            let input = input.to_device(self.device_b);
            let numbers: Vec<f32> = input.to_data().to_vec().unwrap();
            assert_eq!(numbers, numbers_expected);

            // Move tensor back to device 1
            let input = input.to_device(self.device_a);
            let numbers: Vec<f32> = input.to_data().to_vec().unwrap();
            assert_eq!(numbers, numbers_expected);
        }

        fn name(&self) -> String {
            format!("remote")
        }

        fn sync(&self) {
            remote::RemoteBackend::sync(self.device_a);
            remote::RemoteBackend::sync(self.device_b);
        }

        fn shapes(&self) -> Vec<Vec<usize>> {
            vec![self.shape.dims.clone()]
        }
    }

    #[allow(dead_code)]
    pub fn bench<B: burn::backend::BackendIr>(_device: &B::Device) -> Vec<BenchmarkResult> {
        let server_a = LocalServer::<B>::new(3000);
        let server_b = LocalServer::<B>::new(3001);

        let device_a = server_a.get_device();
        let device_b = server_b.get_device();
        let benches = vec![
            RemoteBenchmark::<B>::new(
                Shape {
                    dims: vec![1, 16, 16],
                },
                &device_a,
                &device_b,
            ),
            RemoteBenchmark::<B>::new(
                Shape {
                    dims: vec![1, 8, 8],
                },
                &device_a,
                &device_b,
            ),
        ];

        let mut results = vec![];
        for bench in benches {
            println!("doing bench {:?}", &bench.shape);
            let result = run_benchmark(bench);
            results.push(result);
        }

        println!("shutting down runtimes");

        server_a.runtime.shutdown_background();
        server_b.runtime.shutdown_background();

        results
    }
}

#[cfg(all(
    feature = "test-remote",
    not(feature = "legacy-v16"),
    not(feature = "legacy-v17")
))]
#[allow(dead_code)]
fn bench<B: burn::backend::BackendIr>(device: &B::Device) -> Vec<BenchmarkResult> {
    remote_benchmarks::bench::<B>(device)
}

#[cfg(any(
    not(feature = "test-remote"),
    feature = "legacy-v16",
    feature = "legacy-v17"
))]
#[allow(dead_code)]
fn bench<B: Backend>(_device: &B::Device) -> Vec<BenchmarkResult> {
    vec![]
}

fn main() {
    burnbench::bench_on_backend!()
}
