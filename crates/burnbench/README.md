# Burn Benchmark

This crate allows to compare backend computation times, from tensor operations to complex models.

## burnbench CLI

This crate comes with a CLI binary called `burnbench` which can be executed via
`cargo run --release --bin burnbench`.

Note that you need to run the `release` target of `burnbench` otherwise you won't be able to share
your benchmark results.

The end of options argument `--` is used to pass arguments to the `burnbench` application. For
instance `cargo run --bin burnbench -- list` passes the `list` argument to `burnbench` effectively
calling `burnbench list`.

There is also a cargo alias `cargo bb` which simplifies the command line. The example command above
then becomes: `cargo bb list`.

### Commands

#### List backends

To list all the available backends use the `list` command:

```sh
> cargo run --release --bin burnbench -- list
    Finished dev [unoptimized] target(s) in 0.10s
     Running `target/debug/burnbench list`
Available Backends:
- all
- candle-cpu
- candle-cuda
- candle-metal
- cuda
- cuda-fusion
- rocm
- rocm-fusion
- ndarray
- ndarray-simd
- ndarray-blas-accelerate
- ndarray-blas-netlib
- ndarray-blas-openblas
- tch-cpu
- tch-cuda
- tch-metal
- wgpu
- wgpu-fusion
- vulkan
- vulkan-fusion
- metal
- metal-fusion
```

#### Run benchmarks

To run a given benchmark we use the `run` command with the arguments `--benches` and `--device`. The
device selects the backend at runtime. In the following example we execute the `unary` benchmark on
the `wgpu` device:

```sh
> cargo run --release --bin burnbench -- run --benches unary --device wgpu
```

Shorthands can be used, the following command line is the same:

```sh
> cargo run --release --bin burnbench -- run -b unary -D wgpu
```

Multiple benchmarks and devices can be passed on the same command line. Selecting more devices does
not add builds — they all run on the same binary:

```sh
> cargo run --bin burnbench -- run --benches unary binary --device wgpu cuda
```

Compile-time framework decorators are a separate concern, selected with `--build` (default
`default`). Each profile is its own build, so this is how you compare e.g. fusion on vs off in a
single report:

```sh
> cargo run --bin burnbench -- run --benches matmul --device vulkan --build default no-fusion
```

The number of builds is `benches × build profiles`; devices and dtypes are runtime reruns. Available
build profiles: `default`, `no-fusion`, `no-autotune`, `no-anything`.

By default `burnbench` uses a compact output with a progress bar which hides the compilation logs
and benchmarks results as they are executed. If a benchmark failed to run, the `--verbose` flag can
be used to investigate the error.

#### Authentication and benchmarks sharing

Burnbench can upload benchmark results to our servers so that users can share their results with the
community and we can use this information to drive the development of Burn. The results can be
explored on [Burn website][1].

Sharing results is opt-in and it is enabled with the `--share` arguments passed to the `run`
command:

```sh
> cargo run --release --bin burnbench -- run --share --benches unary --device wgpu
```

To be able to upload results you must be authenticated. We only support GitHub authentication. To
authenticate run the `auth` command, then follow the URL to enter your device code and authorize the
Burnbench application:

```sh
> cargo run --release --bin burnbench -- auth
```

If everything is fine you should get a confirmation in the terminal that your token has been saved
to the burn cache directory.

We don't store any of your personal information. An anonymized user name will be attributed to you
and displayed in the terminal once you are authenticated. For instance:

```
🔑 Your username is: CuteFlame
```

You can now use the `--share` argument to upload and share your benchmarks. A URL to the results
will displayed at the end of the report table.

Note that your access token will be refreshed automatically so you should not need to reauthorize
the application again except if your refresh token itself becomes invalid.

## Execute benchmarks with cargo

Using only cargo, the device is injected at runtime via `--device`, and the compile-time build
profile is selected with cargo features (the `--build` profiles map to feature sets):

```sh
# default profile (fusion + autotune) on the wgpu device
> cargo bench --bench unary -- --device wgpu --dtype f32

# no-fusion profile
> cargo bench --bench unary --no-default-features --features autotune -- --device wgpu
```

## Add a new benchmark

To add a new benchmark it must be first declared in the `Cargo.toml` file of your crate:

```toml
[[bench]]
name = "mybench"
harness = false
```

Create a new file `mybench.rs` in the `benches` directory and implement the `Benchmark` trait over
your benchmark structure. Then implement a `fn bench(device: &Device) -> Vec<BenchmarkResult>`. In
`main`, inject the device and save the results:

```rust
fn main() {
    let device = backend_comparison::select_device();
    let results = bench(&device);
    backend_comparison::save(results, &device);
}
```

## Add a new device

You can register a new device in the `DeviceValues` enumeration and map it to a `Device` in
`backend_comparison::select_device` (in `backend-comparison/src/lib.rs`):

```rs
#[derive(Debug, Clone, PartialEq, Eq, ValueEnum, Display, EnumIter)]
enum DeviceValues {
    // ...
    #[strum(to_string = "mydevice")]
    MyDevice,
    // ...
}
```

[1]: https://burn.dev/benchmarks/community-benchmarks
