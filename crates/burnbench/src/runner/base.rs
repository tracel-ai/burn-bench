use clap::{Parser, Subcommand, ValueEnum};
use percent_encoding::{NON_ALPHANUMERIC, utf8_percent_encode};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitStatus;
use std::sync::{Arc, Mutex};
use strum::{Display, EnumIter, IntoEnumIterator};

use super::auth::Tokens;
use crate::endgroup;
use crate::group;
use crate::runner::workflow::send_output_results;
use crate::runner::workflow::send_started_event;
use crate::system_info::BenchmarkSystemInfo;
use crate::{BENCHMARK_WEBSITE_URL, TRACEL_CI_SERVER_BASE_URL};

use super::auth::get_tokens;
use super::auth::get_username;
use super::dependency::Dependency;
use super::processor::{CargoRunner, NiceProcessor, OutputProcessor, Profiling, VerboseProcessor};
use super::progressbar::RunnerProgressBar;
use super::reports::{BenchmarkCollection, FailedBenchmark};

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
#[derive(Debug)]
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
    #[strum(to_string = "cpu")]
    Cpu,
    #[strum(to_string = "cpu-fusion")]
    CpuFusion,
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

/// Execute burnbench on the provided crate located at the provided path.
pub fn execute<P: AsRef<Path>>(name: &str, path: P) {
    let path: &Path = path.as_ref();
    let info = CrateInfo {
        name: name.to_string(),
        path: path.join("crates").join(name),
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
    match get_tokens()
        .ok_or_else(|| "missing access token".into())
        .and_then(|t| get_username(&t.access_token))
    {
        Ok(user_info) => {
            println!("🔑 Your username is: {}", user_info.nickname);
        }
        Err(e) => {
            eprintln!("❌ Failed to authenticate ({e})");
        }
    }
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
    let inputs_file = std::env::var("WEBHOOK_INPUTS_FILE");
    let emit_started_webhook = std::env::var("BURN_BENCH_EMIT_STARTED_WEBHOOK")
        .ok()
        .map_or(false, |v| v == "true");
    let total_count: u64 = (backends.len() * versions.len() * dtypes.len())
        .try_into()
        .unwrap();
    let runner_pb: Option<Arc<Mutex<RunnerProgressBar>>> = if verbose {
        None
    } else {
        Some(Arc::new(Mutex::new(RunnerProgressBar::new(total_count))))
    };
    // 'started' webhook
    if let Ok(ref inputs) = inputs_file
        && emit_started_webhook
    {
        send_started_event(&inputs);
    }
    // Iterate through every combination of benchmark and backend
    println!("\nBenchmarking Burn @ {versions:?}");
    for version in versions.iter() {
        for backend in backends.iter() {
            for dtype in dtypes.iter() {
                let bench_str = benches.join(", ");
                let backend_str = backend.to_string();
                let url = format!("{TRACEL_CI_SERVER_BASE_URL}benchmarks");

                if verbose {
                    group!("Running benchmarks: {bench_str}@{backend_str}-{dtype}");
                }
                let status = run_cargo(
                    info,
                    benches,
                    &backend_str,
                    dtype,
                    &url,
                    token,
                    &runner_pb,
                    version,
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
                if verbose {
                    endgroup!();
                }
            }
        }
    }

    if let Some(pb) = runner_pb.clone() {
        pb.lock().unwrap().finish();
    }

    let collection = report_collection.load_records();
    let table = collection.get_ascii_table();
    let mut output_results = table.clone();
    let share_link = web_results_url(token, versions);
    if let Some(ref url) = share_link {
        output_results.push_str(&format!("\n\n📊 Browse results at {}", url));
    }
    println!("{output_results}");
    // 'complete' webhook
    if inputs_file.is_ok() {
        send_output_results(&inputs_file.unwrap(), &table, share_link.as_deref());
    }
}

fn get_required_features(info: &CrateInfo, target_bench: &str) -> Vec<String> {
    let cargo_file_path = Path::new(&info.path).join("Cargo.toml");

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
    benches: &[String],
    backend: &str,
    dtype: &BenchDType,
    url: &str,
    token: Option<&str>,
    progress_bar: &Option<Arc<Mutex<RunnerProgressBar>>>,
    version: &str,
    profile: &Profiling,
) -> io::Result<ExitStatus> {
    let bench_str = benches.join(", ");
    let processor: Arc<dyn OutputProcessor> = if let Some(pb) = progress_bar {
        Arc::new(NiceProcessor::new(
            bench_str,
            backend.to_string(),
            version.to_string(),
            pb.clone(),
        ))
    } else {
        Arc::new(VerboseProcessor)
    };
    let dependency_version = get_version(version);
    let dependency = Dependency::new(&dependency_version);
    let mut features = String::new();

    let guard = dependency.patch(info.path.as_path()).unwrap();
    let name = &info.name;
    features += &format!("{name}/{backend},{name}/{dtype}");

    for bench in benches.iter() {
        for req_feature in get_required_features(info, bench) {
            features += &format!(",{}", req_feature);
        }
    }

    if version.starts_with("0.16") {
        features += ",legacy-v16";
    } else if version.starts_with("0.17") {
        features += ",legacy-v17";
    }

    for bench in benches.iter() {
        for req_feature in get_required_features(info, bench) {
            features += &format!(",{name}/{req_feature}");
        }
    }

    let mut args = vec![];
    if benches[0] == "all" {
        args = vec![
            "--benches",
            "--features",
            &features,
            "--target-dir",
            crate::BENCHMARKS_TARGET_DIR,
        ]
    } else {
        for bench in benches.iter() {
            args.push("--bench");
            args.push(bench);
        }
        args.push("--features");
        args.push(&features);
        args.push("--target-dir");
        args.push(crate::BENCHMARKS_TARGET_DIR);
    }

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

/// Take cake of special version names of the form PR#number_sha1 and return sha1.
/// Otherwise just return version untouched.
fn get_version(version: &str) -> String {
    if let Some(suffix) = version.strip_prefix("PR#") {
        if let Some((_, sha)) = suffix.split_once('_') {
            return sha.to_string();
        }
    }
    version.to_string()
}

fn web_results_url(token: Option<&str>, versions: &[String]) -> Option<String> {
    if let Some(t) = token {
        if let Ok(user) = get_username(t) {
            let sysinfo = BenchmarkSystemInfo::new();
            let encoded_os = utf8_percent_encode(&sysinfo.os.name, NON_ALPHANUMERIC).to_string();
            let versions = utf8_percent_encode(&versions.join(","), NON_ALPHANUMERIC).to_string();

            return Some(format!(
                "{}benchmarks/community-benchmarks?user={}&sysHardware=Any&os={}&burnVersions={}",
                BENCHMARK_WEBSITE_URL, user.nickname, encoded_os, versions
            ));
        }
    }
    None
}
