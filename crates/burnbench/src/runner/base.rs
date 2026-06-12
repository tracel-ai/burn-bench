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

    /// Space separated list of devices (backends) to run on. Selecting more
    /// devices does not add builds; they all run on the same binary.
    #[clap(short = 'D', long = "devices", num_args(1..), required = true)]
    devices: Vec<DeviceValues>,

    /// Space separated list of build profiles (compile-time framework
    /// decorators), e.g. `default no-fusion`. Each profile is a separate build.
    /// Defaults to `default`.
    #[clap(long = "builds", num_args(0..))]
    builds: Vec<BuildValues>,

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

/// A backend selected at runtime by injecting the corresponding device. Picking
/// several devices never adds builds: they all run on the same binary.
#[derive(Debug, Clone, PartialEq, Eq, ValueEnum, Display, EnumIter)]
enum DeviceValues {
    #[strum(to_string = "all")]
    All,
    #[cfg(not(target_os = "macos"))]
    #[strum(to_string = "cuda")]
    Cuda,
    #[cfg(target_os = "linux")]
    #[strum(to_string = "rocm")]
    Rocm,
    #[cfg(not(target_os = "macos"))]
    #[strum(to_string = "vulkan")]
    Vulkan,
    #[cfg(target_os = "macos")]
    #[strum(to_string = "metal")]
    Metal,
    #[strum(to_string = "wgpu")]
    Wgpu,
    #[strum(to_string = "webgpu")]
    Webgpu,
    #[strum(to_string = "cpu")]
    Cpu,
    #[strum(to_string = "flex")]
    Flex,
    #[strum(to_string = "ndarray")]
    Ndarray,
    #[strum(to_string = "tch-cpu")]
    TchCpu,
    #[strum(to_string = "tch-cuda")]
    TchCuda,
    #[strum(to_string = "tch-metal")]
    TchMetal,
}

impl DeviceValues {
    /// Whether this device uses the LibTorch backend, which needs the `tch`
    /// cargo feature (and an external libtorch) to be compiled in.
    fn is_tch(&self) -> bool {
        matches!(
            self,
            DeviceValues::TchCpu | DeviceValues::TchCuda | DeviceValues::TchMetal
        )
    }
    fn is_rocm(&self) -> bool {
        matches!(self, DeviceValues::Rocm)
    }
}

/// A compile-time build profile: which framework decorators (`fusion`,
/// `autotune`) are enabled. Each profile is a distinct build, so this is what
/// multiplies the number of builds (together with the selected benches).
///
/// `std` is always enabled because the always-linked GPU backends require it.
#[derive(Debug, Clone, PartialEq, Eq, ValueEnum, Display, EnumIter)]
enum BuildValues {
    /// Everything on: `fusion` + `autotune` (the default features).
    #[strum(to_string = "default")]
    Default,
    /// Disable kernel fusion.
    #[strum(to_string = "no-fusion")]
    NoFusion,
    /// Disable kernel autotuning.
    #[strum(to_string = "no-autotune")]
    NoAutotune,
    /// Disable every framework decorator (raw backend only).
    #[strum(to_string = "no-anything")]
    NoAnything,
}

impl BuildValues {
    /// Maps a profile to its cargo feature configuration: whether to pass
    /// `--no-default-features`, and which framework features to (re-)enable.
    fn features(&self) -> (bool, &'static [&'static str]) {
        match self {
            BuildValues::Default => (false, &[]),
            BuildValues::NoFusion => (true, &["autotune"]),
            BuildValues::NoAutotune => (true, &["fusion"]),
            BuildValues::NoAnything => (true, &[]),
        }
    }
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
    println!("Available devices:");
    for device in DeviceValues::iter() {
        println!("- {}", device);
    }
    println!("\nAvailable build profiles:");
    for build in BuildValues::iter() {
        println!("- {}", build);
    }
}

fn command_run(info: &CrateInfo, mut run_args: RunArgs) {
    let mut tokens: Option<Tokens> = None;
    if run_args.share {
        tokens = get_tokens();
    }
    // Expand `all` to every device except the LibTorch ones: `tch` requires an
    // external libtorch, so it is only pulled in when explicitly requested. Any
    // device the user listed explicitly alongside `all` (e.g. `all tch-cpu`) is
    // preserved. Similarly, `rocm` is only pulled in when explicitly requested.
    let mut devices = run_args.devices.clone();
    if devices.contains(&DeviceValues::All) {
        let explicit: Vec<DeviceValues> = devices
            .iter()
            .filter(|&d| *d != DeviceValues::All)
            .cloned()
            .collect();
        let mut expanded: Vec<DeviceValues> = DeviceValues::iter()
            .filter(|d| *d != DeviceValues::All && !d.is_tch() && !d.is_rocm())
            .collect();
        for device in explicit {
            if !expanded.contains(&device) {
                expanded.push(device);
            }
        }
        devices = expanded;
    }
    let tch_requested = devices.iter().any(DeviceValues::is_tch);
    let rocm_requested = devices.iter().any(DeviceValues::is_rocm);
    let access_token = tokens.map(|t| t.access_token);

    // Set the defaults
    if run_args.builds.is_empty() {
        run_args.builds.push(BuildValues::Default);
    }
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
        &devices,
        &run_args.builds,
        tch_requested,
        rocm_requested,
        &run_args.versions,
        &run_args.dtypes,
        access_token.as_deref(),
        run_args.verbose,
        &profiling,
    );
}

#[allow(clippy::too_many_arguments)]
fn run_backend_comparison_benchmarks(
    info: &CrateInfo,
    benches: &[String],
    devices: &[DeviceValues],
    builds: &[BuildValues],
    tch_requested: bool,
    rocm_requested: bool,
    versions: &[String],
    dtypes: &[BenchDType],
    token: Option<&str>,
    verbose: bool,
    profiling: &Profiling,
) {
    let mut report_collection = BenchmarkCollection::default();
    let inputs_file = std::env::var("WEBHOOK_INPUTS_FILE");
    let total_count: u64 = (versions.len() * builds.len() * devices.len() * dtypes.len())
        .try_into()
        .unwrap();
    let runner_pb: Option<Arc<Mutex<RunnerProgressBar>>> = if verbose {
        None
    } else {
        Some(Arc::new(Mutex::new(RunnerProgressBar::new(total_count))))
    };
    // The build profile (and the set of benches) is what multiplies the number
    // of compilations: for a fixed profile every device/dtype run reuses the
    // same cached binary, since the device and dtype are injected at runtime.
    println!("\nBenchmarking Burn @ {versions:?}");
    for version in versions.iter() {
        for build in builds.iter() {
            for device in devices.iter() {
                for dtype in dtypes.iter() {
                    let bench_str = benches.join(", ");
                    let device_str = device.to_string();
                    let build_str = build.to_string();
                    let label = format!("{device_str} ({build_str})");
                    let url = format!("{TRACEL_CI_SERVER_BASE_URL}benchmarks");

                    if verbose {
                        group!("Running benchmarks: {bench_str}@{label}-{dtype}");
                    }
                    let status = run_cargo(
                        info,
                        benches,
                        &device_str,
                        build,
                        tch_requested,
                        rocm_requested,
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
                        // A failed `cargo bench` invocation fails every bench it
                        // covers; list each failed combination as its own row.
                        for bench in benches.iter() {
                            report_collection.push_failed_benchmark(FailedBenchmark {
                                bench: bench.clone(),
                                version: version.clone(),
                                build: build_str.clone(),
                                device: device_str.clone(),
                                dtype: dtype.to_string(),
                            });
                        }
                    }
                    if verbose {
                        endgroup!();
                    }
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
    if let Ok(inputs_file) = inputs_file {
        send_output_results(&inputs_file, &table, share_link.as_deref());
    }
}

fn get_required_features(info: &CrateInfo, target_bench: &str) -> Vec<String> {
    let cargo_file_path = Path::new(&info.path).join("Cargo.toml");

    let content = fs::read_to_string(&cargo_file_path).expect("Failed to read Cargo.toml");
    let parsed: toml::Table = content.parse().expect("Invalid TOML");

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

#[allow(clippy::too_many_arguments)]
fn run_cargo(
    info: &CrateInfo,
    benches: &[String],
    device: &str,
    build: &BuildValues,
    tch_requested: bool,
    rocm_requested: bool,
    dtype: &BenchDType,
    url: &str,
    token: Option<&str>,
    progress_bar: &Option<Arc<Mutex<RunnerProgressBar>>>,
    version: &str,
    profile: &Profiling,
) -> io::Result<ExitStatus> {
    let bench_str = benches.join(", ");
    let build_str = build.to_string();
    let processor: Arc<dyn OutputProcessor> = if let Some(pb) = progress_bar {
        Arc::new(NiceProcessor::new(
            bench_str,
            format!("{device} ({build_str})"),
            version.to_string(),
            pb.clone(),
        ))
    } else {
        Arc::new(VerboseProcessor)
    };
    let dependency_version = get_version(version);
    let dependency = Dependency::new(&dependency_version);

    let guard = dependency.patch(info.path.as_path()).unwrap();
    let name = &info.name;

    // Backends are selected at runtime by injecting the right device (the
    // `--devices` argument below), so cargo features only control the compile-
    // time build profile (framework decorators), `tch` (when a LibTorch device
    // is requested), and any benchmark-specific required features.
    let (no_default_features, kept_features) = build.features();
    let mut feature_list: Vec<String> = kept_features
        .iter()
        .map(|f| format!("{name}/{f}"))
        .collect();
    if tch_requested {
        feature_list.push(format!("{name}/tch"));
    }
    if rocm_requested {
        feature_list.push(format!("{name}/rocm"));
    }
    for bench in benches.iter() {
        for req_feature in get_required_features(info, bench) {
            feature_list.push(format!("{name}/{req_feature}"));
        }
    }
    let features = feature_list.join(",");
    let dtype_str = dtype.to_string();

    let mut args: Vec<&str> = Vec::new();
    if benches[0] == "all" {
        args.push("--benches");
    } else {
        for bench in benches.iter() {
            args.push("--bench");
            args.push(bench);
        }
    }
    if no_default_features {
        args.push("--no-default-features");
    }
    if !features.is_empty() {
        args.push("--features");
        args.push(&features);
    }
    args.push("--target-dir");
    args.push(crate::BENCHMARKS_TARGET_DIR);

    // Runtime arguments forwarded to the benchmark binary: which device to
    // inject, which dtype to configure, the build label, and optional sharing.
    args.push("--");
    args.push("--devices");
    args.push(device);
    args.push("--dtype");
    args.push(&dtype_str);
    args.push("--builds");
    args.push(&build_str);
    if let Some(t) = token {
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
    if let Some(suffix) = version.strip_prefix("PR#")
        && let Some((_, sha)) = suffix.split_once('_')
    {
        return sha.to_string();
    }
    version.to_string()
}

fn web_results_url(token: Option<&str>, versions: &[String]) -> Option<String> {
    if let Some(t) = token
        && let Ok(user) = get_username(t)
    {
        let sysinfo = BenchmarkSystemInfo::new();
        let encoded_os = utf8_percent_encode(&sysinfo.os.name, NON_ALPHANUMERIC).to_string();
        let versions = utf8_percent_encode(&versions.join(","), NON_ALPHANUMERIC).to_string();

        return Some(format!(
            "{}benchmarks/community-benchmarks?user={}&sysHardware=Any&os={}&burnVersions={}",
            BENCHMARK_WEBSITE_URL, user.nickname, encoded_os, versions
        ));
    }
    None
}
