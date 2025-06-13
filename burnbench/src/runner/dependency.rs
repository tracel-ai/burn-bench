use regex::Regex;
use semver::Version;
use std::io::Write;
use std::time::Duration;
use std::{
    fs::OpenOptions,
    path::{Path, PathBuf},
};

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

pub struct CargoDependencyGuard {
    cargo_file_path: PathBuf,
    original_content: String,
}

impl Drop for CargoDependencyGuard {
    fn drop(&mut self) {
        let mut cargo_file = OpenOptions::new()
            .write(true)
            .open(&self.cargo_file_path)
            .unwrap();
        cargo_file.set_len(0).unwrap();
        write!(cargo_file, "{}", self.original_content).unwrap();
        log::info!("Reset original cargo file");
        std::thread::sleep(Duration::from_millis(200));
    }
}

impl Dependency {
    pub fn patch(&self, base_path: &Path) -> std::io::Result<CargoDependencyGuard> {
        let cargo_file_path = Path::new(base_path).join("Cargo.toml");
        let burn_dir = std::env::var("BURN_BENCH_BURN_DIR").unwrap_or("../../burn/".into());

        let original_content = std::fs::read_to_string(&cargo_file_path)?;

        let content = match self {
            Dependency::Local => self.update_burn_local(&original_content, &burn_dir),
            Dependency::Crate(version) => self.update_burn_version(&original_content, version),
            Dependency::Git(version) => self.update_burn_git(&original_content, version),
        }?;

        let guard = CargoDependencyGuard {
            cargo_file_path: cargo_file_path.to_path_buf(),
            original_content,
        };
        std::fs::write(cargo_file_path, content)?;

        Ok(guard)
    }

    fn update_feature_flags(version: &Version, content: String) -> String {
        if version < &Version::new(0, 17, 0) {
            let content = content
                .replace("cuda = [\"burn/cuda\"]", "cuda = [\"burn/cuda-jit\"]")
                .replace("rocm = [\"burn/rocm\"]", "rocm = [\"burn/hip-jit\"]")
                .replace(
                    "ndarray-simd = [\"ndarray\", \"burn/simd\"]",
                    "ndarray-simd = [\"ndarray\"]",
                )
                .replace(
                    "vulkan = [\"burn/vulkan\", \"burn/autotune\"]",
                    "vulkan = [\"burn/wgpu-spirv\", \"burn/autotune\"]",
                )
                .replace(
                    "metal = [\"burn/vulkan\", \"burn/autotune\"]",
                    "metal = [\"burn/wgpu\", \"burn/autotune\"]",
                )
                .replace(
                    "ndarray-simd = [\"burn/ndarray\", \"burn/simd\"]",
                    "ndarray-simd = [\"burn/ndarray\"]",
                )
                .replace(
                    "candle-metal = [\"burn/candle\", \"burn/candle-metal\"]",
                    "candle-metal = [\"burn/candle\", \"burn/metal\"]",
                )
                // Use matching `rand` version (binary and data benchmarks)
                .replace(
                    "rand = { version = \"0.9.0\" }",
                    "rand = { version = \"0.8.5\" }",
                );

            if (version < &Version::new(0, 16, 1)) & !content.contains("bincode = \"=2.0.0-rc.3\"")
            {
                content.replace(
                    "[dependencies]",
                    "[dependencies]\nbincode = \"=2.0.0-rc.3\"\nbincode_derive = \"=2.0.0-rc.3\"",
                )
            } else {
                content
            }
        } else {
            content
        }
    }
    fn update_burn_version(
        &self,
        content: &str,
        version: &Version,
    ) -> Result<String, std::io::Error> {
        let version_str = version.to_string();
        log::info!("Applying Burn version: {version_str}");

        // Update burn versions
        let burn_re = Regex::new(r"burn = \{ .+ \}").unwrap();

        let content = burn_re
            .replace_all(
                content,
                format!(
                    "burn = {{ version = \"={}\", default-features = false }}",
                    version_str
                ),
            )
            .to_string();

        let content = Self::update_feature_flags(version, content);
        Ok(content)
    }

    // NOTE: [patch] can only be applied at the root of the workspace
    // https://doc.rust-lang.org/cargo/reference/overriding-dependencies.html#the-patch-section
    // Therefore, we apply the change directly to the dependency
    fn update_burn_git(&self, content: &str, reference: &str) -> Result<String, std::io::Error> {
        log::info!("Applying Burn git: {reference}");

        // Update burn git reference
        let burn_re = Regex::new(r"burn = \{ .+ \}").unwrap();

        let content = burn_re.replace_all(content,
            format!("burn = {{ git = \"https://github.com/tracel-ai/burn\", {}, default-features = false }}", reference)
        ).to_string();

        Ok(content)
    }

    fn update_burn_local(&self, content: &str, repo_path: &str) -> Result<String, std::io::Error> {
        log::info!("Applying Burn local: {repo_path}");

        // Update burn path
        let burn_re = Regex::new(r"burn = \{ .+ \}").unwrap();

        let content = burn_re
            .replace_all(
                content,
                format!(
                    "burn = {{ path = \"{}/crates/burn\", default-features = false }}",
                    repo_path
                ),
            )
            .to_string();

        Ok(content)
    }
}

fn is_commit_hash(reference: &str) -> bool {
    // Check if the reference is a valid commit hash (7 to 40 hexadecimal characters)
    let re = Regex::new(r"^[0-9a-f]{7,40}$").unwrap();
    re.is_match(reference)
}
