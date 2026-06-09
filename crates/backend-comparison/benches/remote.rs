use burn::tensor::Device;
use burnbench::BenchmarkResult;

// cargo bb run -b remote --device wgpu -V local

#[cfg(feature = "remote")]
mod remote_benchmarks {
    use burn::tensor::{
        Device, Distribution, Shape, Tensor,
        server::{Channel, start_async},
    };
    use burnbench::{Benchmark, BenchmarkResult, run_benchmark};
    use tokio::runtime::Runtime;

    /// Hosts a backend `device` behind a local WebSocket server, exposing a
    /// matching remote `Device` for clients to connect to.
    struct LocalServer {
        runtime: Runtime,
        device: Device,
    }

    impl LocalServer {
        pub fn new(host: Device, port: u16) -> Self {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_io()
                .build()
                .unwrap();

            runtime.spawn(start_async(host, Channel::WebSocket { port }));

            Self {
                runtime,
                device: Device::remote(&format!("ws://localhost:{port}")),
            }
        }
    }

    struct RemoteBenchmark {
        shape: Shape,
        device_a: Device,
        device_b: Device,
    }

    impl Benchmark for RemoteBenchmark {
        type Input = ();
        type Output = ();

        fn prepare(&self) -> Self::Input {}

        fn execute(&self, _: Self::Input) -> Self::Output {
            // Some random input
            let input =
                Tensor::<3>::random(self.shape.clone(), Distribution::Default, &self.device_a);
            let numbers_expected: Vec<f32> = input.to_data().to_vec().unwrap();

            // Move tensor to device 2
            let input = input.to_device(&self.device_b);
            let numbers: Vec<f32> = input.to_data().to_vec().unwrap();
            assert_eq!(numbers, numbers_expected);

            // Move tensor back to device 1
            let input = input.to_device(&self.device_a);
            let numbers: Vec<f32> = input.to_data().to_vec().unwrap();
            assert_eq!(numbers, numbers_expected);
        }

        fn name(&self) -> String {
            "remote".to_string()
        }

        fn sync(&self) {
            self.device_a.sync().unwrap();
            self.device_b.sync().unwrap();
        }

        fn shapes(&self) -> Vec<Vec<usize>> {
            vec![self.shape.to_vec()]
        }
    }

    #[allow(dead_code)]
    pub fn bench(host: &Device) -> Vec<BenchmarkResult> {
        let server_a = LocalServer::new(host.clone(), 3000);
        let server_b = LocalServer::new(host.clone(), 3001);

        let device_a = server_a.device.clone();
        let device_b = server_b.device.clone();
        let benches = vec![
            RemoteBenchmark {
                shape: [1, 16, 16].into(),
                device_a: device_a.clone(),
                device_b: device_b.clone(),
            },
            RemoteBenchmark {
                shape: [1, 8, 8].into(),
                device_a,
                device_b,
            },
        ];

        let mut results = vec![];
        for bench in benches {
            let result = run_benchmark(bench);
            results.push(result);
        }

        server_a.runtime.shutdown_background();
        server_b.runtime.shutdown_background();

        results
    }
}

#[cfg(feature = "remote")]
#[allow(dead_code)]
fn bench(host: &Device) -> Vec<BenchmarkResult> {
    remote_benchmarks::bench(host)
}

#[cfg(not(feature = "remote"))]
#[allow(dead_code)]
fn bench(_host: &Device) -> Vec<BenchmarkResult> {
    vec![]
}

fn main() {
    let device = backend_comparison::select_device();
    let results = bench(&device);
    backend_comparison::save(results, &device);
}
