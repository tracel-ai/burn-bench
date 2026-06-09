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
                workspace: Some(update(content)),
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
        let benches = self.benches.as_ref().map(|_| TomlDependencyGuard {
            cargo_file_path: content.benches_path.clone(),
            original_content: content.benches.clone(),
        });

        let workspace = self.workspace.as_ref().map(|_| TomlDependencyGuard {
            cargo_file_path: content.workspace_path.clone().unwrap(),
            original_content: content.workspace.clone().unwrap(),
        });

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
        let burn_dir = std::env::var("BURN_BENCH_BURN_DIR").unwrap_or("../../burn/".into());
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

    fn update_burn_version(
        &self,
        content: &DependencyContent,
        version: &Version,
    ) -> Result<DependencyContentUpdate, std::io::Error> {
        let version_str = version.to_string();
        log::info!("Applying Burn version: {version_str}");

        // Update burn versions
        let update = |content: &str| {
            rewrite_burn_deps(content, |_base| format!("version = \"={version_str}\""))
        };

        Ok(content.update(update))
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
            rewrite_burn_deps(content, |_base| {
                format!("git = \"https://github.com/tracel-ai/burn\", {reference}")
            })
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
            let repo_path = repo_path.as_path();
            rewrite_burn_deps(content, |base| {
                format!("path = \"{}crates/{base}\"", repo_path.to_str().unwrap())
            })
        };

        Ok(content.update(update))
    }
}

fn is_commit_hash(reference: &str) -> bool {
    // Check if the reference is a valid commit hash (7 to 40 hexadecimal characters)
    let re = Regex::new(r"^[0-9a-f]{7,40}$").unwrap();
    re.is_match(reference)
}

/// Rewrites the source specifier (git/path/version) of every `burn*` dependency
/// block, preserving any `features = [...]` array already declared on that block.
///
/// Backends are now selected through features declared directly on the `burn`
/// dependency (including OS-conditional target tables), so the patch must keep
/// them instead of collapsing every block to a feature-less `default-features =
/// false` declaration.
fn rewrite_burn_deps(content: &str, source_for: impl Fn(&str) -> String) -> String {
    let features_re = Regex::new(r"features\s*=\s*\[[^\]]*\]").unwrap();
    let mut content = content.to_string();
    for base in BURN_BASE {
        // Anchor the dependency name to the start of a line so commented-out
        // template lines (`# burn = { ... }`) are left untouched.
        let regex = format!("(?m)^{base}{REGEX_BASE}");
        let burn_re = Regex::new(&regex).unwrap();
        content = burn_re
            .replace_all(&content, |caps: &regex::Captures| {
                let matched = &caps[0];
                let features = features_re
                    .find(matched)
                    .map(|m| format!(", {}", m.as_str()))
                    .unwrap_or_default();
                format!(
                    "{base} = {{ {}, default-features = false{features} }}",
                    source_for(base)
                )
            })
            .to_string();
    }
    content
}
