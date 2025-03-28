# Burn Benchmarks

`burn-bench` is a benchmarking repository for [Burn](https://github.com/tracel-ai/burn). It helps
track performance across different hardware and software configurations, making it easier to
identify regressions, improvements, and the best backend for a given workload.

## Structure

- **`backend-comparison/`**: Benchmarks for backend performance, ranging from individual tensor
  operations to full forward and backward passes for a given model.
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

To compare performance across versions:

```sh
cd backend-comparison
./compare.sh 0.16.0 main --benches unary binary --backends  wgpu wgpu-fusion
```

For detailed instructions, see the [`backend-comparison/README.md`](./backend-comparison/README.md).

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
