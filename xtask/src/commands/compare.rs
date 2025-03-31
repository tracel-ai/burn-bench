use regex::Regex;
use semver::Version;
use std::{collections::HashMap, path::{Path, PathBuf}};
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
        let verbose = if self.verbose { "" } else { " -v" };
        format!(
            "--benches {} --backends {}{verbose}",
            self.benches.join(" "),
            self.backends.join(" ")
        )
    }
}

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
        log::info!("UPDATE VERSION {version}");
        let content = match Version::parse(version) {
            Ok(version) => {
                let content = self.read_content()?;
                let mut content = self.update_burn_version(&content, &version)?;

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
                content
            }
            Err(_) => {
                let content = self.read_content()?;
                
                let git_ref = if is_commit_hash(&version) {
                    format!("rev = \"{version}\"")
                } else {
                    format!("branch = \"{version}\"")
                };

                let content = self.update_burn_git(&content, &git_ref)?;
                log::warn!("Assuming version >= 0.17 for git commit, you may need to manually check the cuda feature flag name.");
                self.replace_feature_flags_ge_17(&content)
            },
        };

        self.write_content(content)?;
        log::info!("{} updated successfully with version: {}", Self::SOURCE, version);
        Ok(())
    }

    fn read_content(&self) -> Result<String, std::io::Error> {
        std::fs::read_to_string(self.source())
    }

    fn write_content(&self, content: String) -> Result<(), std::io::Error> {
        std::fs::write(self.source(), content)
    }

    fn update_burn_version(&self, content: &str, version: &Version) -> Result<String, std::io::Error> {
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

        let content = burn_re.replace_all(&content, 
            format!("burn = {{ git = \"https://github.com/tracel-ai/burn\", {}, default-features = false }}", reference)
        ).to_string();
        
        let content = burn_common_re.replace_all(&content, 
            format!("burn-common = {{ git = \"https://github.com/tracel-ai/burn\", {} }}", reference)
        ).to_string();

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
    }
}

fn is_commit_hash(reference: &str) -> bool {
    // Check if the reference is a valid commit hash (7 to 40 hexadecimal characters)
    let re = Regex::new(r"^[0-9a-f]{7,40}$").unwrap();
    re.is_match(reference)
}

/// Burn bench info.
pub(crate) struct BurnBench {
    version: String,
    args: String,
}

impl BurnBenchCompareArgs {
    pub(crate) fn parse(self) -> anyhow::Result<()> {
        // TODO:  BURN BENCHMARK EXECUTION SUMMARY
        /* Do you want to proceed? ([y]/n):

        Created backup of Cargo.toml at Cargo.toml.bak
        Applying Burn version: 0.16.0
        Version < 0.17.0 detected, changing feature flags
        Cargo.toml updated successfully with version: 0.16.0
        ----------------------------------------------
        Running burnbench for version: 0.16.0
        Command: cargo run --release --bin burnbench -- run --benches unary --backends ndarray
        ----------------------------------------------
        */

        log::info!("========================================================");
        log::info!("           BURN BENCHMARK EXECUTION SUMMARY             ");
        log::info!("========================================================");
        log::info!("Version to benchmark:");
        for (i, version) in self.versions.iter().enumerate() {
            log::info!("  {}. {}", i + 1, version);
        }

        let manifest = BenchManifest {
            root: "./backend-comparison".into(),
        };

        manifest.create_backup()?;

        for version in self.versions {
            manifest.update(&version)?;
            BurnBench::run(version, &self.run_args)?;
            manifest.restore_backup()?;
        }
        log::info!("All benchmark runs completed!");
        Ok(())
    }
}

impl BurnBench {
    pub(crate) fn run(version: String, run_args: &BurnBenchRunArgs) -> anyhow::Result<()> {
        let bench = BurnBench {
            version,
            args: run_args.to_string(),
        };
        bench.execute()
    }

    fn execute(&self) -> anyhow::Result<()> {
        log::info!("Running burnbench for version: {}", self.version);
        run_bench(
            // "cargo bb run",
            &self.args.split(" ").collect::<Vec<_>>(),
            None,
            None,
            // Some(self.path),
            "backend comparison should run successfully",
        )
    }
}



/// Run a process
pub fn run_bench(
    args: &[&str],
    envs: Option<HashMap<&str, &str>>,
    path: Option<&Path>,
    error_msg: &str,
) -> anyhow::Result<()> {
    let cmd = "cargo bb run";
    let joined_args = args.join(" ");
    group_info!("Command line: {} {}", cmd, &joined_args);
    let mut command = std::process::Command::new(cmd);
    if let Some(path) = path {
        command.current_dir(path);
    }
    if let Some(envs) = envs {
        command.envs(&envs);
    }
    // command.args(args).output()
    let output = command.args(args).output().map_err(|e| {
        anyhow::anyhow!(
            "Failed to execute {} {}: {}",
            cmd,
            args.first().unwrap(),
            e
        )
    })?;
    if !output.status.success() {
        return Err(anyhow::anyhow!("{}", error_msg));
    }

    let stdout = String::from_utf8(output.stdout)?;
    println!("BURN BENCH OUTPUT:\n{}", stdout);

    anyhow::Ok(())
}