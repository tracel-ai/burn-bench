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

#[allow(dead_code)] // Used to keep things.
pub struct CargoDependencyGuard {
    benches: Option<TomlDependencyGuard>,
    workspace: Option<TomlDependencyGuard>,
}

struct TomlDependencyGuard {
    cargo_file_path: PathBuf,
    original_content: String,
}

struct DependencyContent {
    benches: String,
    benches_path: PathBuf,
    workspace: Option<String>,
    workspace_path: Option<PathBuf>,
}

static BURN_BASE: [&str; 3] = ["burn", "burn-common", "burn-import"];
// Match any char except \} including new lines.
static REGEX_BASE: &str = r" = \{([^\}]|\n)*\}";

impl DependencyContent {
    fn update<F: FnOnce(&str) -> String>(&self, update: F) -> DependencyContentUpdate {
        match &self.workspace {
            Some(content) => DependencyContentUpdate {
                benches: None,
                workspace: Some(update(&content)),
            },
            None => DependencyContentUpdate {
                benches: Some(update(&self.benches)),
                workspace: None,
            },
        }
    }
}

struct DependencyContentUpdate {
    benches: Option<String>,
    workspace: Option<String>,
}

impl DependencyContentUpdate {
    fn create_guard(&self, content: &DependencyContent) -> CargoDependencyGuard {
        let benches = match self.benches {
            Some(_) => Some(TomlDependencyGuard {
                cargo_file_path: content.benches_path.clone(),
                original_content: content.benches.clone(),
            }),
            None => None,
        };

        let workspace = match self.workspace {
            Some(_) => Some(TomlDependencyGuard {
                cargo_file_path: content.workspace_path.clone().unwrap(),
                original_content: content.workspace.clone().unwrap(),
            }),
            None => None,
        };

        CargoDependencyGuard { benches, workspace }
    }

    fn perform_update(&self, content: &DependencyContent) -> std::io::Result<()> {
        if let Some(updated) = &self.benches {
            std::fs::write(&content.benches_path, updated)?;
        }

        if let Some(updated) = &self.workspace {
            std::fs::write(content.workspace_path.as_ref().unwrap(), updated)?;
        };

        Ok(())
    }
}

impl DependencyContent {
    pub fn from_path(base_path: &Path) -> std::io::Result<Self> {
        let benches_path = Path::new(base_path).join("Cargo.toml");
        let benches = std::fs::read_to_string(&benches_path)?;
        let mut workspace = None;
        let mut workspace_path = None;

        let mut burn_in_workspace = false;

        if benches.contains("burn = \"workspace\"") {
            burn_in_workspace = true;
        }
        if benches.contains("burn = { workspace = true") {
            burn_in_workspace = true;
        }

        if burn_in_workspace {
            let cargo_file_path = Path::new(".").join("Cargo.toml");
            let content = std::fs::read_to_string(&cargo_file_path)?;
            workspace = Some(content);
            workspace_path = Some(cargo_file_path);
        }

        Ok(Self {
            benches,
            benches_path,
            workspace,
            workspace_path,
        })
    }
}

impl Drop for TomlDependencyGuard {
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
        let burn_dir = std::env::var("BURN_BENCH_BURN_DIR").unwrap_or("../burn/".into());
        let content_original = DependencyContent::from_path(base_path)?;

        let content = match self {
            Dependency::Local => self.update_burn_local(&content_original, &burn_dir),
            Dependency::Crate(version) => self.update_burn_version(&content_original, version),
            Dependency::Git(version) => self.update_burn_git(&content_original, version),
        }?;

        let guard = content.create_guard(&content_original);
        content.perform_update(&content_original)?;

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
        content: &DependencyContent,
        version: &Version,
    ) -> Result<DependencyContentUpdate, std::io::Error> {
        let version_str = version.to_string();
        log::info!("Applying Burn version: {version_str}");

        // Update burn versions

        let update_version = |content: &str| {
            let mut content = content.to_string();
            for base in BURN_BASE {
                let regex = base.to_string() + REGEX_BASE;
                let burn_re = Regex::new(&regex).unwrap();
                content = burn_re
                    .replace_all(
                        content.as_str(),
                        format!(
                            "{base} = {{ version = \"={}\", default-features = false }}",
                            version_str
                        ),
                    )
                    .to_string();
            }

            content
        };

        match &content.workspace {
            Some(original) => {
                let workspace = update_version(&original);
                let benches = Self::update_feature_flags(version, content.benches.clone());

                Ok(DependencyContentUpdate {
                    benches: Some(benches),
                    workspace: Some(workspace),
                })
            }
            None => {
                let benches = update_version(&content.benches);
                let benches = Self::update_feature_flags(version, benches);

                Ok(DependencyContentUpdate {
                    benches: Some(benches),
                    workspace: None,
                })
            }
        }
    }

    // NOTE: [patch] can only be applied at the root of the workspace
    // https://doc.rust-lang.org/cargo/reference/overriding-dependencies.html#the-patch-section
    // Therefore, we apply the change directly to the dependency
    fn update_burn_git(
        &self,
        content: &DependencyContent,
        reference: &str,
    ) -> Result<DependencyContentUpdate, std::io::Error> {
        log::info!("Applying Burn git: {reference}");

        // Update burn git reference
        let update = |content: &str| {
            let mut content = content.to_string();
            for base in BURN_BASE {
                let regex = base.to_string() + REGEX_BASE;
                let burn_re = Regex::new(&regex).unwrap();
                content = burn_re.replace_all(
                    content.as_str(),
                    format!("{base} = {{ git = \"https://github.com/tracel-ai/burn\", {}, default-features = false }}", reference)
                ).to_string();
            }

            content
        };

        Ok(content.update(update))
    }

    fn update_burn_local(
        &self,
        content: &DependencyContent,
        repo_path: &str,
    ) -> Result<DependencyContentUpdate, std::io::Error> {
        log::info!("Applying Burn local: {repo_path}");

        // Update burn path
        let repo_path = match content.workspace_path {
            Some(_) => Path::new(repo_path).to_path_buf(),
            None => Path::new("../").join(repo_path),
        };
        let update = |content: &str| {
            let mut content = content.to_string();
            let repo_path = repo_path.as_path();

            for base in BURN_BASE {
                let regex = base.to_string() + REGEX_BASE;
                let burn_re = Regex::new(&regex).unwrap();
                content = burn_re
                    .replace_all(
                        &content,
                        format!(
                            "{base} = {{ path = \"{}crates/{base}\", default-features = false }}",
                            repo_path.to_str().unwrap()
                        ),
                    )
                    .to_string();
            }

            content
        };

        Ok(content.update(update))
    }
}

fn is_commit_hash(reference: &str) -> bool {
    // Check if the reference is a valid commit hash (7 to 40 hexadecimal characters)
    let re = Regex::new(r"^[0-9a-f]{7,40}$").unwrap();
    re.is_match(reference)
}
