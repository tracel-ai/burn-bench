use glob::glob;
use tracel_xtask::prelude::*;

#[derive(clap::Args)]
pub struct ProfileArgs {
    #[command(flatten)]
    command: BenchOptionsArgs,
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum CudaBackend {
    Cubecl,
    CubeclFusion,
    Libtorch,
}

#[derive(clap::Args, Debug)]
pub(crate) struct BenchOptionsArgs {
    #[clap(short = 'b', long = "bench", required = true)]
    pub bench: String,
    #[clap(long)]
    #[clap(short = 'B', long = "backend", required = true)]
    pub backend: CudaBackend,
    #[arg(long, default_value = "ncu")]
    pub ncu_path: String,
    #[arg(long, default_value = "ncu-ui")]
    pub ncu_ui_path: String,
}

pub(crate) struct Profile {}

impl ProfileArgs {
    pub(crate) fn run(&self) -> anyhow::Result<()> {
        Profile::run(&self.command)
    }
}

impl Profile {
    pub(crate) fn run(args: &BenchOptionsArgs) -> anyhow::Result<()> {
        Profile {}.execute(args)
    }

    fn execute(&self, command: &BenchOptionsArgs) -> anyhow::Result<()> {
        ensure_cargo_crate_is_installed("mdbook", None, None, false)?;
        group!("Profile: {:?}", command);
        self.bench(command)?;
        endgroup!();
        Ok(())
    }

    fn bench(&self, options: &BenchOptionsArgs) -> anyhow::Result<()> {
        let get_benches = |bench: &str| {
            let pattern = format!("./target/release/deps/{}-*", bench);
            let files: Vec<_> = glob(&pattern)
                .into_iter()
                .flat_map(|r| r.filter_map(|f| f.ok()))
                .collect();

            files
        };

        get_benches(&options.bench)
            .into_iter()
            .for_each(|f| std::fs::remove_file(f).unwrap());

        let feature = match options.backend {
            CudaBackend::Cubecl => "cuda",
            CudaBackend::CubeclFusion => "cuda-fusion",
            CudaBackend::Libtorch => "tch-gpu",
        };

        run_process(
            "cargo",
            &[
                "build",
                "--bench",
                &options.bench,
                "--release",
                "--features",
                feature,
            ],
            None,
            None,
            "Can build bench.",
        )?;

        let bins = get_benches(&options.bench);
        let bin = bins.first().unwrap().as_path().to_str().unwrap();
        let file = format!("target/{}", options.bench);

        let ncu_bin_path = std::process::Command::new("which")
            .arg(&options.ncu_path)
            .output()
            .map_err(|_| ())
            .and_then(|output| String::from_utf8(output.stdout).map_err(|_| ()))
            .expect("Can't find ncu. Make sure it is installed and in your PATH.");

        run_process(
            "sudo",
            &[
                "BENCH_NUM_SAMPLES=1",
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
            format!("Should profile {}", options.bench).as_str(),
        )?;

        let output = format!("{}.ncu-rep", file);
        run_process(
            &options.ncu_ui_path,
            &[&output],
            None,
            None,
            format!("Should open results for {}", options.bench).as_str(),
        )
    }
}
