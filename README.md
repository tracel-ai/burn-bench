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

To benchmark performance across version(s):

```sh
cargo bench-versions 0.16.0 main local --benches unary --backends wgpu-fusion
```

You can specify one or more versions and provide custom `burnbench` arguments to benchmark them.

```sh
cargo bench-versions <version1> [version2...] <burnbench_args>
```

This will run benchmarks on the specified versions and log the results in a timestamped file,
allowing you to compare their performance.

The versions can be one of:

- Published version (e.g., `0.16.0`)
- Git branch (e.g., `main`)
- Git commit hash
- `local`

By default, the `local` version points to a relative path for the Burn repo directory (`../../burn`
relative to `backend-comparison/`). This can be modified via the `BURN_DIR` environment variable.

> **Note:** this might not work out of the box for previous versions with unspecified breaking
> changes to the API or feature flag names. We currently handle changes after the 0.16 release.  
> To handle feature flag changes, you probably want to modify
> [`compare.rs`](./xtask/src/commands/compare.rs) to overwrite the `Cargo.toml` based on some
> condition. See for example
> [`replace_feature_flags_lt_0_17`](./xtask/src/commands/compare.rs#318).  
> For breaking API changes, this can be handled in the build script to add a cfg. See for example
> [`burn_version_lt_0170`](./backend-comparison/build.rs#L372) and how it is
> [used for conditional compilation](./backend-comparison/src/persistence/base.rs#L71).

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
