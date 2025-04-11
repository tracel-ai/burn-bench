use super::progressbar::RunnerProgressBar;
use glob::glob;
use std::collections::HashMap;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

/// Processor for standard output of cargo process
pub trait OutputProcessor: Send + Sync + 'static {
    /// Process a line
    fn process_line(&self, line: &str);
    /// To be called to indicate progress has been made
    fn progress(&self);
    /// To be called went the processor has finished processing
    fn finish(&self);
}

/// A processor that does nothing except printing the output lines as is.
#[derive(Default)]
pub struct VerboseProcessor;

impl OutputProcessor for VerboseProcessor {
    fn process_line(&self, line: &str) {
        println!("{}", line);
    }
    fn progress(&self) {}
    fn finish(&self) {}
}

/// A processor that just send the output into oblivion.
#[derive(Default)]
pub struct SinkProcessor;

impl OutputProcessor for SinkProcessor {
    fn process_line(&self, _line: &str) {}
    fn progress(&self) {}
    fn finish(&self) {}
}

/// A processor for a nice and compact output experience using a progress bar
pub struct NiceProcessor {
    bench: String,
    backend: String,
    pb: Arc<Mutex<RunnerProgressBar>>,
}

pub(crate) enum NiceProcessorState {
    Default,
    Compiling,
    Running,
    Uploading,
}

impl NiceProcessor {
    pub fn new(bench: String, backend: String, pb: Arc<Mutex<RunnerProgressBar>>) -> Self {
        Self { bench, backend, pb }
    }

    pub fn format_pb_message(&self, state: NiceProcessorState) -> String {
        match state {
            NiceProcessorState::Default | NiceProcessorState::Compiling => {
                format!("🔨 {} ▶ {}", self.bench, self.backend)
            }
            NiceProcessorState::Running => {
                format!("🔥 {} ▶ {}", self.bench, self.backend)
            }
            NiceProcessorState::Uploading => {
                format!("💾 {} ▶ {}", self.bench, self.backend)
            }
        }
    }
}

impl OutputProcessor for NiceProcessor {
    fn process_line(&self, line: &str) {
        let pb = self.pb.lock().unwrap();
        let state = if line.contains("Compiling") {
            pb.stop_spinner();
            NiceProcessorState::Compiling
        } else if line.contains("Running") {
            pb.stop_spinner();
            NiceProcessorState::Running
        } else if line.contains("Sharing") {
            pb.start_spinner();
            NiceProcessorState::Uploading
        } else {
            NiceProcessorState::Default
        };
        pb.message(self.format_pb_message(state));
    }

    fn progress(&self) {
        self.pb.lock().unwrap().advance_spinner();
    }

    fn finish(&self) {
        self.pb.lock().unwrap().inc_by_one();
    }
}

/// Benchmark runner using cargo bench.
pub struct CargoRunner<'a> {
    params: &'a [&'a str],
    envs: Vec<(String, String)>,
    processor: Arc<dyn OutputProcessor>,
    profiling: Profiling,
}

#[derive(Clone)]
pub enum Profiling {
    Deactivated,
    Activated {
        ncu_path: String,
        ncu_ui_path: String,
    },
}

impl<'a> CargoRunner<'a> {
    fn run_profile(&self, ncu_path: &str, ncu_ui_path: &str) -> io::Result<ExitStatus> {
        let get_benches = |bench: &str| {
            let pattern = format!("./target/benchmarks/release/deps/{}-*", bench);
            let files: Vec<_> = glob(&pattern)
                .into_iter()
                .flat_map(|r| r.filter_map(|f| f.ok()))
                .collect();

            files
        };

        let bench = &self.params[1];
        log::info!("Profiling benchmark {bench:?}");
        get_benches(bench)
            .into_iter()
            .for_each(|f| std::fs::remove_file(f).unwrap());

        let cargo = Command::new("cargo")
            .env("CARGO_TERM_COLOR", "always")
            .envs(self.envs.iter().map(|(k, v)| (k, v)))
            .arg("build")
            .arg("--release")
            .args(self.params)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Cargo command should start successfully");

        self.run_command(cargo)?;

        let bins = get_benches(&bench);
        let bin = bins.first().unwrap().as_path().to_str().unwrap();
        let file = format!("target/{}", bench);

        let ncu_bin_path = std::process::Command::new("which")
            .arg(ncu_path)
            .output()
            .map_err(|_| ())
            .and_then(|output| String::from_utf8(output.stdout).map_err(|_| ()))
            .expect("Can't find ncu. Make sure it is installed and in your PATH.");

        let libtorch = std::env::var("LIBTORCH").unwrap_or_default();
        let ld = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();

        let libtorch_env = format!("LIBTORCH={libtorch}");
        let ld_env = format!("LD_LIBRARY_PATH={ld}");

        run_process(
            "sudo",
            &[
                "BENCH_NUM_SAMPLES=1",
                &libtorch_env,
                &ld_env,
                ncu_bin_path.trim(),
                "--nvtx",
                "--set=full",
                "--call-stack",
                "--export",
                &file,
                "--force-overwrite",
                bin,
            ],
            None,
            None,
        )?;

        let output = format!("{}.ncu-rep", file);
        run_process(ncu_ui_path, &[&output], None, None)
    }

    pub fn new(
        params: &'a [&'a str],
        envs: Vec<(String, String)>,
        processor: Arc<dyn OutputProcessor>,
        profiling: Profiling,
    ) -> Self {
        Self {
            params,
            envs,
            processor,
            profiling,
        }
    }

    pub fn run(&self) -> io::Result<ExitStatus> {
        match &self.profiling {
            Profiling::Deactivated => self.run_bench(),
            Profiling::Activated {
                ncu_path,
                ncu_ui_path,
            } => self.run_profile(&ncu_path, &ncu_ui_path),
        }
    }

    fn run_command(&self, mut cargo: Child) -> io::Result<ExitStatus> {
        // stdout
        let stdout = BufReader::new(cargo.stdout.take().expect("stdout should be captured"));
        let stdout_processor = Arc::clone(&self.processor);
        let stdout_thread = thread::spawn(move || {
            for line in stdout.lines() {
                let line = line.expect("A line from stdout should be read");
                stdout_processor.process_line(&line);
                stdout_processor.progress();
            }
        });
        // stderr
        let stderr = BufReader::new(cargo.stderr.take().expect("stderr should be captured"));
        let stderr_processor = Arc::clone(&self.processor);
        let stderr_thread = thread::spawn(move || {
            for line in stderr.lines() {
                let line = line.expect("A line from stderr should be read");
                stderr_processor.process_line(&line);
                stderr_processor.progress();
            }
        });
        // wait for process completion
        stdout_thread
            .join()
            .expect("The stderr thread should not panic");
        stderr_thread
            .join()
            .expect("The stderr thread should not panic");
        self.processor.finish();
        cargo.wait()
    }
    fn run_bench(&self) -> io::Result<ExitStatus> {
        let cargo = Command::new("cargo")
            .env("CARGO_TERM_COLOR", "always")
            .envs(self.envs.iter().map(|(k, v)| (k, v)))
            .arg("bench")
            .args(self.params)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Cargo command should start successfully");

        self.run_command(cargo)
    }
}

fn run_process(
    name: &str,
    args: &[&str],
    envs: Option<HashMap<&str, &str>>,
    path: Option<&Path>,
) -> io::Result<ExitStatus> {
    let joined_args = args.join(" ");
    log::info!("Command line: {} {}", name, &joined_args);
    let mut command = Command::new(name);
    if let Some(path) = path {
        command.current_dir(path);
    }
    if let Some(envs) = envs {
        command.envs(&envs);
    }
    let status = command.args(args).status().unwrap();

    Ok(status)
}
