use std::{pin::Pin, time::Duration};

use crate::{BenchmarkComputations, BenchmarkDurations, BenchmarkResult, TimingMethod};

/// Benchmark trait.
pub trait Benchmark {
    /// Benchmark input arguments.
    type Input: Clone;
    /// The benchmark output.
    type Output;

    /// Prepare the benchmark, run anything that is essential for the benchmark, but shouldn't
    /// count as included in the duration.
    ///
    /// # Notes
    ///
    /// This should not include warmup, the benchmark will be run at least one time without
    /// measuring the execution time.
    fn prepare(&self) -> Self::Input;

    /// Execute the benchmark and returns the logical output of the task executed.
    ///
    /// It is important to return the output since otherwise deadcode optimization might optimize
    /// away code that should be benchmarked.
    fn execute(&self, input: Self::Input) -> Self::Output;

    /// Number of samples per run required to have a statistical significance.
    fn num_samples(&self) -> usize {
        const DEFAULT: usize = 10;

        std::env::var("BENCH_NUM_SAMPLES")
            .map(|val| str::parse::<usize>(&val).unwrap_or(DEFAULT))
            .unwrap_or(DEFAULT)
    }

    /// Name of the benchmark, should be short and it should match the name
    /// defined in the crate Cargo.toml
    fn name(&self) -> String;

    /// The options passed to the benchmark.
    fn options(&self) -> Option<String> {
        None
    }

    /// Shapes dimensions
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![]
    }

    /// Wait for computation to complete.
    fn sync(&self);

    /// Start measuring the computation duration.
    fn profile(&self, args: Self::Input) -> ProfileDuration {
        self.profile_full(args)
    }

    /// Start measuring the computation duration. Use the full duration irregardless of whether
    /// device duration is available or not.
    fn profile_full(&self, args: Self::Input) -> ProfileDuration {
        self.sync();
        let start_time = std::time::Instant::now();
        let out = self.execute(args);
        self.sync();
        core::mem::drop(out);
        ProfileDuration::from_duration(start_time.elapsed())
    }

    /// Run the benchmark a number of times.
    #[allow(unused_variables)]
    fn run(&self, timing_method: TimingMethod) -> BenchmarkDurations {
        let execute = |args: &Self::Input| {
            let profile = match timing_method {
                TimingMethod::System => self.profile_full(args.clone()),
                TimingMethod::Device => self.profile(args.clone()),
            };
            futures_lite::future::block_on(profile.resolve())
        };
        let args = self.prepare();

        // Warmup
        for _ in 0..3 {
            let _duration = execute(&args);
        }
        std::thread::sleep(Duration::from_secs(1));

        // Real execution.
        let mut durations = Vec::with_capacity(self.num_samples());
        for _ in 0..self.num_samples() {
            durations.push(execute(&args));
        }

        BenchmarkDurations {
            timing_method,
            durations,
        }
    }
}

/// Result from profiling between two measurements. This can either be a duration or a future that resolves to a duration.
pub enum ProfileDuration {
    /// Client profile contains a full duration.
    Full(Duration),
    /// Client profile measures the device duration, and requires to be resolved.
    DeviceDuration(Pin<Box<dyn Future<Output = Duration> + Send + 'static>>),
}

impl core::fmt::Debug for ProfileDuration {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ProfileDuration::Full(duration) => write!(f, "Full({:?})", duration),
            ProfileDuration::DeviceDuration(_) => write!(f, "DeviceDuration"),
        }
    }
}

impl ProfileDuration {
    /// Create a new `ProfileDuration` straight from a duration.
    pub fn from_duration(duration: Duration) -> Self {
        ProfileDuration::Full(duration)
    }

    /// Create a new `ProfileDuration` from a future that resolves to a duration.
    pub fn from_future(future: impl Future<Output = Duration> + Send + 'static) -> Self {
        ProfileDuration::DeviceDuration(Box::pin(future))
    }

    /// The method used to measure the execution time.
    pub fn timing_method(&self) -> TimingMethod {
        match self {
            ProfileDuration::Full(_) => TimingMethod::System,
            ProfileDuration::DeviceDuration(_) => TimingMethod::Device,
        }
    }

    /// Resolve the actual duration of the profile, possibly by waiting for the future to complete.
    pub async fn resolve(self) -> Duration {
        match self {
            ProfileDuration::Full(duration) => duration,
            ProfileDuration::DeviceDuration(future) => future.await,
        }
    }
}

/// Runs the given benchmark on the device and prints result and information.
pub fn run_benchmark<BM>(benchmark: BM) -> BenchmarkResult
where
    BM: Benchmark,
{
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .unwrap();
    let git_hash = String::from_utf8(output.stdout).unwrap().trim().to_string();
    let durations = benchmark.run(TimingMethod::System);

    BenchmarkResult {
        raw: durations.clone(),
        computed: BenchmarkComputations::new(&durations),
        git_hash,
        name: benchmark.name(),
        options: benchmark.options(),
        shapes: benchmark.shapes(),
        timestamp,
    }
}
