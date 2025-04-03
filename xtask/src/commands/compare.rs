use regex::Regex;
use semver::Version;
use std::env;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use tracel_xtask::prelude::*;

#[derive(clap::Args)]
#[command(about = "Compare multiple Burn versions using burnbench")]
pub struct BurnBenchCompareArgs {
    /// One or more Burn versions, git branches, or commit hashes
    #[arg(required = true)]
    pub versions: Vec<String>,

    /// The burn bench run arguments.
    #[command(flatten)]
    pub run_args: BurnBenchRunArgs,
}

#[derive(clap::Args, Debug)]
pub struct BurnBenchRunArgs {
    /// Enable verbose mode
    #[clap(short = 'v', long = "verbose")]
    verbose: bool,

    /// Space separated list of backends to include
    #[clap(short = 'B', long = "backends", value_name = "BACKEND BACKEND ...", num_args(1..), required = true)]
    backends: Vec<String>,

    /// Space separated list of benches to run
    #[clap(short = 'b', long = "benches", value_name = "BENCH BENCH ...", num_args(1..), required = true)]
    benches: Vec<String>,
}

impl ToString for BurnBenchRunArgs {
    fn to_string(&self) -> String {
        let verbose = if self.verbose { " -v" } else { "" };
        format!(
            "--benches {} --backends {}{verbose}",
            self.benches.join(" "),
            self.backends.join(" ")
        )
    }
}

impl BurnBenchCompareArgs {
    pub(crate) fn parse(self) -> anyhow::Result<()> {
        let mut log_file = LogFile::new()?;

        // Log execution summary to file & log output
        let mut log_both = |msg: &str| -> anyhow::Result<()> {
            log_file.log_line(msg)?;
            log::info!("{msg}");
            Ok(())
        };

        log_both("========================================================")?;
        log_both("           BURN BENCHMARK EXECUTION SUMMARY             ")?;
        log_both("========================================================")?;
        log_both("Running burnbench with:")?;
        log_both(&format!("  {}", self.run_args.to_string()))?;
        log_both("Version to benchmark:")?;
        for (i, version) in self.versions.iter().enumerate() {
            log_both(&format!("  {}. {}", i + 1, version))?;
        }
        log_both("========================================================")?;

        let manifest = BenchManifest {
            root: "./backend-comparison".into(),
        };

        self.run(&manifest, &mut log_file).map_err(|err| {
            manifest.restore_backup().ok();
            err
        })?;
        Ok(())
    }

    fn run(self, manifest: &BenchManifest, log_file: &mut LogFile) -> anyhow::Result<()> {
        manifest.create_backup()?;

        for version in self.versions {
            manifest.update(&version)?;
            BurnBench::run_with_log(version, &self.run_args, log_file)?;
            manifest.restore_backup()?;
        }
        log::info!("All benchmark runs completed!");
        log::info!("Check out the aggregated results: {}", log_file.path());
        Ok(())
    }
}

/// Burn bench info.
pub(crate) struct BurnBench {
    version: String,
    args: String,
}

impl BurnBench {
    pub(crate) fn run_with_log(
        version: String,
        run_args: &BurnBenchRunArgs,
        log_file: &mut LogFile,
    ) -> anyhow::Result<()> {
        let bench = BurnBench {
            version,
            args: run_args.to_string(),
        };
        bench.execute(Some(log_file))
    }

    fn execute(&self, log_file: Option<&mut LogFile>) -> anyhow::Result<()> {
        log::info!("Running burnbench for version: {}", self.version);
        run_bench(
            &self.args.split(" ").collect::<Vec<_>>(),
            log_file,
            "backend comparison should run successfully",
        )
    }
}

pub(crate) enum Dependency {
    Local,
    Crate(Version),
    Git(String),
}

impl Dependency {
    pub fn new(version: &str) -> Self {
        if version == "local" {
            Self::Local
        } else if let Ok(version) = Version::parse(version) {
            Self::Crate(version)
        } else {
            let git_ref = if is_commit_hash(version) {
                format!("rev = \"{version}\"")
            } else {
                format!("branch = \"{version}\"")
            };
            Self::Git(git_ref)
        }
    }
}

/// Cargo.toml manifest updates operations for benchmark comparison.
pub(crate) struct BenchManifest {
    root: PathBuf,
}

impl BenchManifest {
    const SOURCE: &'static str = "Cargo.toml";
    const BACKUP: &'static str = "Cargo.toml.bak";

    pub(crate) fn source(&self) -> PathBuf {
        self.root.join(Self::SOURCE)
    }

    pub(crate) fn backup(&self) -> PathBuf {
        self.root.join(Self::BACKUP)
    }

    pub(crate) fn create_backup(&self) -> Result<(), std::io::Error> {
        std::fs::copy(self.source(), self.backup())?;
        log::info!("Created backup of {} at {}", Self::SOURCE, Self::BACKUP);
        Ok(())
    }

    pub(crate) fn restore_backup(&self) -> Result<(), std::io::Error> {
        std::fs::copy(self.backup(), self.source())?;
        log::info!("Restored backup at {}", Self::SOURCE);
        Ok(())
    }

    pub(crate) fn update(&self, version: &str) -> Result<(), std::io::Error> {
        let burn_dir = env::var("BURN_DIR").unwrap_or("../../burn".to_string());
        let mut content = self.read_content()?;

        match Dependency::new(version) {
            Dependency::Local => {
                content = self.update_burn_local(&content, &burn_dir)?;
                log::warn!(
                    "Assuming version >= 0.17 for local repo, you may need to manually check the cuda feature flag name."
                );
                content = self.replace_feature_flags_ge_17(&content);
            }
            Dependency::Crate(version) => {
                content = self.update_burn_version(&content, &version)?;
                if version < Version::new(0, 17, 0) {
                    log::info!("Version < 0.17.0 detected, changing feature flags");
                    content = self.replace_feature_flags_lt_0_17(content);
                    // Pin bincode pre-release (used in burn < 0.17)
                    if !content.contains("bincode = \"=2.0.0-rc.3\"") {
                        content = content.replace(
                                "[dependencies]",
                                "[dependencies]\nbincode = \"=2.0.0-rc.3\"\nbincode_derive = \"=2.0.0-rc.3\""
                            );
                    }
                } else {
                    log::info!(
                        "Version >= 0.17.0 detected, using cuda, hip, vulkan and simd feature flags"
                    );
                    content = self.replace_feature_flags_ge_17(&content);
                }
            }
            Dependency::Git(git_ref) => {
                content = self.update_burn_git(&content, &git_ref)?;
                log::warn!(
                    "Assuming version >= 0.17 for git commit, you may need to manually check the cuda feature flag name."
                );
                content = self.replace_feature_flags_ge_17(&content);
            }
        };

        self.write_content(content)?;
        log::info!(
            "{} updated successfully with version: {}",
            Self::SOURCE,
            version
        );
        Ok(())
    }

    fn read_content(&self) -> Result<String, std::io::Error> {
        std::fs::read_to_string(self.source())
    }

    fn write_content(&self, content: String) -> Result<(), std::io::Error> {
        std::fs::write(self.source(), content)
    }

    fn update_burn_version(
        &self,
        content: &str,
        version: &Version,
    ) -> Result<String, std::io::Error> {
        let version_str = version.to_string();
        log::info!("Applying Burn version: {version_str}");

        // Update burn and burn-common versions
        let burn_re = Regex::new(r"burn = \{ .+, default-features = false \}").unwrap();
        let burn_common_re = Regex::new(r"burn-common = \{ .+ \}").unwrap();

        let content = burn_re
            .replace_all(
                content,
                format!(
                    "burn = {{ version = \"{}\", default-features = false }}",
                    version_str
                ),
            )
            .to_string();

        let content = burn_common_re
            .replace_all(
                &content,
                format!("burn-common = {{ version = \"{}\" }}", version_str),
            )
            .to_string();

        Ok(content)
    }

    fn update_burn_git(&self, content: &str, reference: &str) -> Result<String, std::io::Error> {
        log::info!("Applying Burn git: {reference}");

        // Update burn and burn-common git reference
        let burn_re = Regex::new(r"burn = \{ .+, default-features = false \}").unwrap();
        let burn_common_re = Regex::new(r"burn-common = \{ .+ \}").unwrap();

        let content = burn_re.replace_all(content,
            format!("burn = {{ git = \"https://github.com/tracel-ai/burn\", {}, default-features = false }}", reference)
        ).to_string();

        let content = burn_common_re
            .replace_all(
                &content,
                format!(
                    "burn-common = {{ git = \"https://github.com/tracel-ai/burn\", {} }}",
                    reference
                ),
            )
            .to_string();

        Ok(content)
    }

    fn update_burn_local(&self, content: &str, repo_path: &str) -> Result<String, std::io::Error> {
        log::info!("Applying Burn local: {repo_path}");

        // Update burn and burn-common path
        let burn_re = Regex::new(r"burn = \{ .+, default-features = false \}").unwrap();
        let burn_common_re = Regex::new(r"burn-common = \{ .+ \}").unwrap();

        let content = burn_re
            .replace_all(
                content,
                format!(
                    "burn = {{ path = \"{}/crates/burn\", default-features = false }}",
                    repo_path
                ),
            )
            .to_string();

        let content = burn_common_re
            .replace_all(
                &content,
                format!(
                    "burn-common = {{ path = \"{}/crates/burn-common\" }}",
                    repo_path
                ),
            )
            .to_string();

        Ok(content)
    }

    fn replace_feature_flags_lt_0_17(&self, content: String) -> String {
        content
            .replace("cuda = [\"burn/cuda\"]", "cuda = [\"burn/cuda-jit\"]")
            .replace("hip = [\"burn/hip\"]", "hip = [\"burn/hip-jit\"]")
            .replace(
                "wgpu-spirv = [\"burn/vulkan\", \"burn/autotune\"]",
                "wgpu-spirv = [\"burn/wgpu-spirv\", \"burn/autotune\"]",
            )
            .replace(
                "ndarray-simd = [\"burn/ndarray\", \"burn/simd\"]",
                "ndarray-simd = [\"burn/ndarray\"]",
            )
            // Use matching `rand` version (binary and data benchmarks)
            .replace(
                "rand = { version = \"0.9.0\" }",
                "rand = { version = \"0.8.5\" }",
            )
    }

    fn replace_feature_flags_ge_17(&self, content: &str) -> String {
        content
            .replace("cuda = [\"burn/cuda-jit\"]", "cuda = [\"burn/cuda\"]")
            .replace("hip = [\"burn/hip-jit\"]", "hip = [\"burn/hip\"]")
            .replace(
                "wgpu-spirv = [\"burn/wgpu-spirv\", \"burn/autotune\"]",
                "wgpu-spirv = [\"burn/vulkan\", \"burn/autotune\"]",
            )
            .replace(
                "ndarray-simd = [\"burn/ndarray\"]",
                "ndarray-simd = [\"burn/ndarray\", \"burn/simd\"]",
            )
            // Use matching `rand` version (binary and data benchmarks)
            .replace(
                "rand = { version = \"0.8.5\" }",
                "rand = { version = \"0.9.0\" }",
            )
    }
}

fn is_commit_hash(reference: &str) -> bool {
    // Check if the reference is a valid commit hash (7 to 40 hexadecimal characters)
    let re = Regex::new(r"^[0-9a-f]{7,40}$").unwrap();
    re.is_match(reference)
}

pub(crate) fn run_bench(
    args: &[&str],
    log_file: Option<&mut LogFile>,
    error_msg: &str,
) -> anyhow::Result<()> {
    let full_args = [&["bb", "run"], args].concat();
    let cmd = "cargo";
    let joined_args = full_args.join(" ");
    log::info!("Command line: {} {}", cmd, &joined_args);

    let mut child = Command::new(cmd)
        .env("CARGO_TERM_COLOR", "always")
        .args(&full_args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| anyhow::anyhow!("Failed to execute cargo {}: {}", full_args.join(" "), e))?;

    let mut log_file = log_file.as_ref().map(|x| x.try_clone().unwrap());

    // stdout
    let stdout = BufReader::new(child.stdout.take().expect("stdout should be captured"));
    let stdout_thread = std::thread::spawn(move || {
        for line in stdout.lines() {
            let line = line.expect("A line from stdout should be read");
            println!("{line}");

            if let Some(ref mut file) = log_file {
                file.log_line(&line).unwrap();
            }
        }
    });

    // stderr
    let stderr = BufReader::new(child.stderr.take().expect("stderr should be captured"));
    let stderr_thread = std::thread::spawn(move || {
        for line in stderr.lines() {
            let line = line.expect("A line from stderr should be read");
            eprintln!("{line}");
        }
    });

    // Wait for the process to complete
    let status = child.wait()?;
    stdout_thread
        .join()
        .expect("The stderr thread should not panic");
    stderr_thread
        .join()
        .expect("The stderr thread should not panic");

    if !status.success() {
        return Err(anyhow::anyhow!("{}", error_msg));
    }
    anyhow::Ok(())
}

pub(crate) struct LogFile {
    file: File,
    path: String,
}

impl LogFile {
    pub fn new() -> anyhow::Result<Self> {
        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
        let path = format!("burnbench_{}.log", timestamp);

        let file = OpenOptions::new().create(true).append(true).open(&path)?;

        Ok(LogFile { file, path })
    }

    pub fn log_line(&mut self, line: &str) -> anyhow::Result<()> {
        writeln!(self.file, "{}", line)?;
        self.file.flush()?;
        Ok(())
    }

    pub fn path(&self) -> &str {
        self.path.as_ref()
    }

    // Clone the file handle for use in a different thread
    pub fn try_clone(&self) -> anyhow::Result<Self> {
        let file = self.file.try_clone()?;

        Ok(LogFile {
            file,
            path: self.path.clone(),
        })
    }
}
