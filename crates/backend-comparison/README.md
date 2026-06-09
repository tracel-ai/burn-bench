# Backend Comparison

This crate defines a set of benchmarks to run with `burnbench`.

## Run benchmarks

To run a given benchmark we use the `run` command with the arguments `--benches` and `--device`. The
device selects the backend at runtime. In the following example we execute the `unary` benchmark on
the `wgpu` device:

```sh
> cargo bb run --benches unary --device wgpu
```

Shorthands can be used, the following command line is the same:

```sh
> cargo bb -- run -b unary -D wgpu
```

Selecting several devices does **not** add builds — they all run on the same binary:

```sh
> cargo bb run --benches unary --device wgpu cuda vulkan
```

Compile-time framework decorators are selected with `--build` (default `default`). This is how you
compare, for example, fusion on vs off — each profile is a separate build, but both appear in the
same comparison table:

```sh
> cargo bb run --benches matmul --device vulkan --build default no-fusion
```

Available build profiles: `default`, `no-fusion`, `no-autotune`, `no-anything`. The number of builds
is `benches × build profiles`; devices and dtypes are runtime reruns of those builds.

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
> cargo bb run --share --benches unary --device wgpu
```

To be able to upload results you must be authenticated. We only support GitHub authentication. To
authenticate run the `auth` command, then follow the URL to enter your device code and authorize the
Burnbench application:

```sh
> cargo bb auth
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

Backends are no longer selected through cargo features: every backend that can be compiled on the
host is linked in automatically, and the concrete backend is chosen at runtime by passing
`--device` (and optionally `--dtype`) to the benchmark binary.

```sh
# Run the unary benchmark on the wgpu backend in f32
> cargo bench --bench unary -- --device wgpu --dtype f32
```

Kernel fusion is a compile-time decorator, so the fused variants require the `fusion` feature:

```sh
> cargo bench --bench unary --features fusion -- --device wgpu
```

Backends that depend on external libraries are opt-in features: `tch` (LibTorch) and the BLAS
ndarray variants (`ndarray-blas-accelerate`, `ndarray-blas-netlib`, `ndarray-blas-openblas`,
`ndarray-simd`).

## Add a new benchmark

To add a new benchmark it must be first declared in the `Cargo.toml` file of this crate:

```toml
[[bench]]
name = "mybench"
harness = false
```

Create a new file `mybench.rs` in the `benches` directory and implement the `Benchmark` trait over
your benchmark structure. Then implement a `fn bench(device: &Device) -> Vec<BenchmarkResult>`
function. Finally, in `main`, inject the device and save the results:

```rust
fn main() {
    let device = backend_comparison::select_device();
    let results = bench(&device);
    backend_comparison::save(results, &device);
}
```

For multi-device benchmarks, use `backend_comparison::select_devices()` to obtain every available
device of the selected backend.

[1]: https://burn.dev/benchmarks/community-benchmarks
