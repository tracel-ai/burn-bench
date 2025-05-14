use clap::{Parser, Subcommand, ValueEnum};
use percent_encoding::{NON_ALPHANUMERIC, utf8_percent_encode};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitStatus;
use std::sync::{Arc, Mutex};
use strum::{Display, EnumIter, IntoEnumIterator};

use super::auth::Tokens;
use crate::system_info::BenchmarkSystemInfo;

use super::auth::get_tokens;
use super::auth::get_username;
use super::dependency::Dependency;
use super::processor::{CargoRunner, NiceProcessor, OutputProcessor, Profiling, VerboseProcessor};
use super::progressbar::RunnerProgressBar;
use super::reports::{BenchmarkCollection, FailedBenchmark};
use crate::USER_BENCHMARK_WEBSITE_URL;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Authenticate using GitHub
    Auth,
    /// List all available backends
    List,
    /// Runs benchmarks
    Run(RunArgs),
}

/// Information about the crate to benchmark.
struct CrateInfo {
    /// The name of the crate that contains the benchmarks.
    name: String,
    /// The path from which the command burnbench will be run.
    path: PathBuf,
}

#[derive(Parser, Debug)]
struct RunArgs {
    /// Share the benchmark results by uploading them to Burn servers
    #[clap(short = 's', long = "share")]
    share: bool,

    /// Enable verbose mode
    #[clap(short = 'v', long = "verbose")]
    verbose: bool,

    /// Space separated list of backends to include
    #[clap(short = 'B', long = "backends", num_args(1..), required = true)]
    backends: Vec<BackendValues>,

    /// Space separated list of benches to run
    #[clap(short = 'b', long = "benches", num_args(0..))]
    benches: Vec<String>,

    /// One or more Burn versions, git branches, or commit hashes
    ///
    /// Default using @main.
    #[clap(short = 'V', long = "versions", num_args(0..))]
    pub versions: Vec<String>,

    #[clap(short = 'd', long = "dtypes", num_args(0..))]
    pub dtypes: Vec<BenchDType>,

    #[clap(short = 'p', long = "profile", default_value = "false")]
    pub profile: bool,

    #[arg(long, default_value = "ncu")]
    pub ncu_path: String,
    #[arg(long, default_value = "ncu-ui")]
    pub ncu_ui_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq, ValueEnum, Display, EnumIter)]
enum BenchDType {
    #[strum(to_string = "f32")]
    F32,
    #[strum(to_string = "f16")]
    F16,
    #[strum(to_string = "flex32")]
    FLEX32,
    #[strum(to_string = "bf16")]
    BF16,
}

#[derive(Debug, Clone, PartialEq, Eq, ValueEnum, Display, EnumIter)]
enum BackendValues {
    #[strum(to_string = "all")]
    All,
    #[strum(to_string = "candle-cpu")]
    CandleCpu,
    #[strum(to_string = "candle-cuda")]
    CandleCuda,
    #[strum(to_string = "candle-metal")]
    CandleMetal,
    #[strum(to_string = "cuda")]
    Cuda,
    #[strum(to_string = "cuda-fusion")]
    CudaFusion,
    #[cfg(target_os = "linux")]
    #[strum(to_string = "rocm")]
    Rocm,
    #[cfg(target_os = "linux")]
    #[strum(to_string = "rocm-fusion")]
    RocmFusion,
    #[strum(to_string = "ndarray")]
    Ndarray,
    #[strum(to_string = "ndarray-simd")]
    NdarraySimd,
    #[strum(to_string = "ndarray-blas-accelerate")]
    NdarrayBlasAccelerate,
    #[strum(to_string = "ndarray-blas-netlib")]
    NdarrayBlasNetlib,
    #[strum(to_string = "ndarray-blas-openblas")]
    NdarrayBlasOpenblas,
    #[strum(to_string = "tch-cpu")]
    TchCpu,
    #[strum(to_string = "tch-cuda")]
    TchCuda,
    #[strum(to_string = "tch-metal")]
    TchMetal,
    #[strum(to_string = "wgpu")]
    Wgpu,
    #[strum(to_string = "wgpu-fusion")]
    WgpuFusion,
    #[strum(to_string = "vulkan")]
    Vulkan,
    #[strum(to_string = "vulkan-fusion")]
    VulkanFusion,
    #[strum(to_string = "metal")]
    Metal,
    #[strum(to_string = "metal-fusion")]
    MetalFusion,
}

/// Execute burnbench on the provided crate localted at the provided path.
pub fn execute<P: AsRef<Path>>(name: &str, path: P) {
    let path: &Path = path.as_ref();
    let info = CrateInfo {
        name: name.to_string(),
        path: path.join(name),
    };
    let args = Args::parse();
    match args.command {
        Commands::Auth => command_auth(),
        Commands::List => command_list(),
        Commands::Run(run_args) => command_run(&info, run_args),
    }
}

/// Create an access token from GitHub Burnbench application, store it,
/// and display the name of the authenticated user.
fn command_auth() {
    get_tokens()
        .and_then(|t| get_username(&t.access_token))
        .map(|user_info| {
            println!("🔑 Your username is: {}", user_info.nickname);
        })
        .unwrap_or_else(|| {
            println!("Failed to display your username.");
        });
}

fn command_list() {
    println!("Available Backends:");
    for backend in BackendValues::iter() {
        println!("- {}", backend);
    }
}

fn command_run(info: &CrateInfo, mut run_args: RunArgs) {
    let mut tokens: Option<Tokens> = None;
    if run_args.share {
        tokens = get_tokens();
    }
    // collect benchmarks and benches to execute
    let mut backends = run_args.backends.clone();
    if backends.contains(&BackendValues::All) {
        backends = BackendValues::iter()
            .filter(|b| b != &BackendValues::All)
            .collect();
    }
    let access_token = tokens.map(|t| t.access_token);

    // Set the defaults
    if run_args.dtypes.is_empty() {
        run_args.dtypes.push(BenchDType::F32);
    }
    if run_args.benches.is_empty() {
        run_args.benches.push("all".to_string());
    }
    if run_args.versions.is_empty() {
        run_args.versions.push("main".to_string());
    }

    let profiling = if run_args.profile {
        Profiling::Activated {
            ncu_path: run_args.ncu_path,
            ncu_ui_path: run_args.ncu_ui_path,
        }
    } else {
        Profiling::Deactivated
    };
    run_backend_comparison_benchmarks(
        info,
        &run_args.benches,
        &backends,
        &run_args.versions,
        &run_args.dtypes,
        access_token.as_deref(),
        run_args.verbose,
        &profiling,
    );
}

fn run_backend_comparison_benchmarks(
    info: &CrateInfo,
    benches: &[String],
    backends: &[BackendValues],
    versions: &[String],
    dtypes: &[BenchDType],
    token: Option<&str>,
    verbose: bool,
    profiling: &Profiling,
) {
    let mut report_collection = BenchmarkCollection::default();
    let total_count: u64 = (backends.len() * benches.len() * versions.len() * dtypes.len())
        .try_into()
        .unwrap();
    let runner_pb: Option<Arc<Mutex<RunnerProgressBar>>> = if verbose {
        None
    } else {
        Some(Arc::new(Mutex::new(RunnerProgressBar::new(total_count))))
    };
    // Iterate through every combination of benchmark and backend
    println!("\nBenchmarking Burn @ {versions:?}");
    for version in versions.iter() {
        for backend in backends.iter() {
            for bench in benches.iter() {
                for dtype in dtypes.iter() {
                    let bench_str = bench.to_string();
                    let backend_str = backend.to_string();
                    let url = format!("{}benchmarks", crate::USER_BENCHMARK_SERVER_URL);

                    let status = run_cargo(
                        info,
                        &bench_str,
                        &backend_str,
                        dtype,
                        &url,
                        token,
                        &runner_pb,
                        &version,
                        profiling,
                    );
                    let success = status.unwrap().success();

                    if success {
                        if let Some(ref pb) = runner_pb {
                            pb.lock().unwrap().succeeded_inc();
                        }
                    } else {
                        if let Some(ref pb) = runner_pb {
                            pb.lock().unwrap().failed_inc();
                        }
                        report_collection.push_failed_benchmark(FailedBenchmark {
                            bench: bench_str.clone(),
                            backend: backend_str.clone(),
                        })
                    }
                }
            }
        }
    }

    if let Some(pb) = runner_pb.clone() {
        pb.lock().unwrap().finish();
    }

    println!("{}", report_collection.load_records());
    if let Some(url) = web_results_url(token) {
        println!("📊 Browse results at {}", url);
    }
}

fn get_required_features(target_bench: &str) -> Vec<String> {
    let cargo_file_path = Path::new("backend-comparison").join("Cargo.toml");

    let content = fs::read_to_string(&cargo_file_path).expect("Failed to read Cargo.toml");
    let parsed: toml::Value = content.parse().expect("Invalid TOML");

    let benches = parsed.get("bench").and_then(|b| b.as_array()).unwrap();

    for bench in benches {
        let name = bench
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        if name == target_bench {
            if let Some(features) = bench.get("required-features").and_then(|f| f.as_array()) {
                let feature_list: Vec<String> = features
                    .iter()
                    .filter_map(|v| v.as_str())
                    .map(String::from)
                    .collect();
                return feature_list;
            }
            return vec![];
        }
    }

    vec![]
}

fn run_cargo(
    info: &CrateInfo,
    bench: &str,
    backend: &str,
    dtype: &BenchDType,
    url: &str,
    token: Option<&str>,
    progress_bar: &Option<Arc<Mutex<RunnerProgressBar>>>,
    version: &str,
    profile: &Profiling,
) -> io::Result<ExitStatus> {
    let processor: Arc<dyn OutputProcessor> = if let Some(pb) = progress_bar {
        Arc::new(NiceProcessor::new(
            bench.to_string(),
            backend.to_string(),
            pb.clone(),
        ))
    } else {
        Arc::new(VerboseProcessor)
    };
    let dependency = Dependency::new(version);
    let guard = dependency.patch(info.path.as_path()).unwrap();
    let name = &info.name;
    let mut features = format!("{name}/{backend},{name}/{dtype}");

    if version.starts_with("0.16") {
        features += ",legacy-v16";
    } else if version.starts_with("0.17") {
        features += ",legacy-v17";
    }

    for req_feature in get_required_features(bench) {
        features += &format!(",{}", req_feature);
    }

    let mut args = if bench == "all" {
        vec![
            "--benches",
            "--features",
            &features,
            "--target-dir",
            crate::BENCHMARKS_TARGET_DIR,
        ]
    } else {
        vec![
            "--bench",
            bench,
            "--features",
            &features,
            "--target-dir",
            crate::BENCHMARKS_TARGET_DIR,
        ]
    };

    if let Some(t) = token {
        args.push("--");
        args.push("--sharing-url");
        args.push(url);
        args.push("--sharing-token");
        args.push(t);
    }
    let runner = CargoRunner::new(
        &args,
        vec![("BURN_BENCH_BURN_VERSION".to_string(), version.to_string())],
        processor,
        profile.clone(),
    );
    let status = runner.run();

    core::mem::drop(guard);

    status
}

fn web_results_url(token: Option<&str>) -> Option<String> {
    if let Some(t) = token {
        if let Some(user) = get_username(t) {
            let sysinfo = BenchmarkSystemInfo::new();
            let encoded_os = utf8_percent_encode(&sysinfo.os.name, NON_ALPHANUMERIC).to_string();
            let output = std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .output()
                .unwrap();
            let git_hash = String::from_utf8(output.stdout).unwrap().trim().to_string();

            return Some(format!(
                "{}benchmarks/community-benchmarks?user={}&os={}&version1={}&version2={}&search=true",
                USER_BENCHMARK_WEBSITE_URL, user.nickname, encoded_os, git_hash, git_hash
            ));
        }
    }
    None
}
