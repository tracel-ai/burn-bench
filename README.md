# Burn Benchmarks

`burn-bench` is a benchmarking repository for [Burn](https://github.com/tracel-ai/burn). It helps
track performance across different hardware and software configurations, making it easier to
identify regressions, improvements, and the best backend for a given workload.

## Structure

- **`backend-comparison/`**: Benchmarks for backend performance, ranging from individual tensor
  operations to full forward and backward passes for a given model.
- **`burnbench/`**: The core benchmarking crate and CLI. Can be used as a standalone tool or
  integrated as a library to define and run custom benchmark suites.
- **(Future)** **`integration-tests/`**: TBD. We'd like to add more tests to capture more complex
  workloads, including evaluation of model convergence, metrics, and overall training performance.

## Getting Started

To run backend performance benchmarks, use the `burnbench` CLI:

```sh
cargo run --release --bin burnbench -- run --benches unary --backends wgpu-fusion
```

Or use the shorthand alias:

```sh
cargo bb run -b unary -B wgpu-fusion
```

This will use the main branch of Burn by default.

To benchmark performance across version(s):

```sh
cargo bb run -b unary -B wgpu-fusion -V 0.18.0 main local
```

You can specify one or more versions and provide custom `burnbench` arguments to benchmark them.

The versions can be one of:

- Published version (e.g., `0.18.0`)
- Git branch (e.g., `main`)
- Git commit hash
- `local`

By default, the `local` version points to a relative path for the Burn repo directory (`../../burn`
relative to `backend-comparison/`). This can be modified via the `BURN_BENCH_BURN_DIR` environment
variable.

For detailed instructions, see [`burnbench/README.md`](./burnbench/README.md) and
[`backend-comparison/README.md`](./backend-comparison/README.md).

## Community Benchmarks

Burn supports sharing benchmark results to help users compare hardware and backend performance.
Results are published at [burn.dev/benchmarks](https://burn.dev/benchmarks/community-benchmarks).

To contribute benchmarks, authenticate using:

```sh
cargo run --release --bin burnbench -- auth
```

Then share results with:

```sh
cargo bb run --share --benches unary --backends wgpu-fusion
```

## Contributing

We welcome contributions to improve benchmarking coverage and add new performance tests.
