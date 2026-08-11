use std::{pin::Pin, time::Duration};

use crate::{BenchmarkComputations, BenchmarkDurations, BenchmarkResult, Limit, TimingMethod};

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
    ///
    /// When [Benchmark::num_inputs()] is greater than one, this is called once per input, and
    /// each call should return a *distinct* input (e.g. freshly allocated tensors) so that
    /// executions don't all hit the same cached buffers.
    fn prepare(&self) -> Self::Input;

    /// Execute the benchmark and returns the logical output of the task executed.
    ///
    /// It is important to return the output since otherwise deadcode optimization might optimize
    /// away code that should be benchmarked.
    fn execute(&self, input: Self::Input) -> Self::Output;

    /// Number of samples per run required to have a statistical significance.
    fn num_samples(&self) -> usize {
        const DEFAULT: usize = 15;

        std::env::var("BENCH_NUM_SAMPLES")
            .map(|val| str::parse::<usize>(&val).unwrap_or(DEFAULT))
            .unwrap_or(DEFAULT)
    }

    /// How long the benchmark should be executed before starting to measure, to let the device
    /// reach a steady state (clock ramp, caches, lazy initialization).
    ///
    /// This is a *duration*, not an iteration count: short kernels are executed many more times
    /// than long ones, so every benchmark gets the same amount of ramp-up. The warmup phase always
    /// runs at least one execution, even when the duration is zero.
    ///
    /// Can be overridden with the `BENCH_WARMUP_MS` environment variable.
    fn warmup(&self) -> Duration {
        const DEFAULT_MS: u64 = 200;

        let millis = std::env::var("BENCH_WARMUP_MS")
            .map(|val| str::parse::<u64>(&val).unwrap_or(DEFAULT_MS))
            .unwrap_or(DEFAULT_MS);

        Duration::from_millis(millis)
    }

    /// Number of distinct inputs to prepare and cycle through during the run, execution `i` using
    /// input `i % num_inputs`.
    ///
    /// The default of one reuses a single input for every execution, which lets it stay resident
    /// in cache and makes memory-bound kernels look faster than they can be in production. Pick a
    /// value such that `num_inputs * working set` comfortably exceeds the last level cache to
    /// measure cold-ish traffic instead.
    ///
    /// Only used when [Benchmark::prepare_cloned()] is true; otherwise [Benchmark::prepare()] is
    /// already called before every execution.
    fn num_inputs(&self) -> usize {
        1
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

    /// Best-case resource usage of this problem: its memory traffic and a list of
    /// typed compute descriptors (arithmetic / tensor-core, with dtype and MMA
    /// tile). Used to compute utilization against the measured practical limits.
    ///
    /// The default is empty, which leaves the utilization columns as `N/A`.
    fn limits(&self) -> Limit {
        Limit::default()
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
        let execute = |args: Self::Input| {
            let profile = match timing_method {
                TimingMethod::System => self.profile_full(args),
                TimingMethod::Device => self.profile(args),
            };
            futures_lite::future::block_on(profile.resolve())
        };

        let mut durations = Vec::with_capacity(self.num_samples());
        let warmup = self.warmup();

        if self.prepare_cloned() {
            // Distinct inputs, cycled through so that a single working set doesn't simply stay
            // resident in cache for the whole run.
            let num_inputs = self.num_inputs().max(1);
            let inputs: Vec<Self::Input> = (0..num_inputs).map(|_| self.prepare()).collect();

            // Warmup, for a duration rather than a fixed number of iterations, so that short
            // kernels get as much device ramp-up as long ones.
            let start = std::time::Instant::now();
            let mut iteration = 0;
            loop {
                let _duration = execute(inputs[iteration % num_inputs].clone());
                iteration += 1;

                if start.elapsed() >= warmup {
                    break;
                }
            }

            // Real execution.
            for sample in 0..self.num_samples() {
                durations.push(execute(inputs[sample % num_inputs].clone()));
            }
        } else {
            // Warmup
            let start = std::time::Instant::now();
            loop {
                let _duration = execute(self.prepare());

                if start.elapsed() >= warmup {
                    break;
                }
            }

            // Real execution.
            for _ in 0..self.num_samples() {
                durations.push(execute(self.prepare()));
            }
        }

        BenchmarkDurations {
            timing_method,
            durations,
        }
    }

    /// When true, [Benchmark::prepare()] is called only once and the inputs are reused for all
    /// execution using [Clone::clone].
    ///
    /// When false, [Benchmark::prepare()] is called before every execution.
    fn prepare_cloned(&self) -> bool {
        true
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
        limit: benchmark.limits(),
        timestamp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    struct TestBenchmark {
        warmup: Duration,
        num_inputs: usize,
        num_samples: usize,
        prepare_cloned: bool,
        execution_time: Duration,
        /// Number of calls to [Benchmark::prepare()], also used as input id.
        prepared: RefCell<usize>,
        /// Ids of the inputs given to [Benchmark::execute()], in order.
        executed: RefCell<Vec<usize>>,
    }

    impl Default for TestBenchmark {
        fn default() -> Self {
            Self {
                warmup: Duration::ZERO,
                num_inputs: 1,
                num_samples: 3,
                prepare_cloned: true,
                execution_time: Duration::ZERO,
                prepared: RefCell::new(0),
                executed: RefCell::new(Vec::new()),
            }
        }
    }

    impl TestBenchmark {
        /// Ids of the inputs used by the measured samples, skipping the warmup ones.
        fn sampled(&self) -> Vec<usize> {
            let executed = self.executed.borrow();
            executed[executed.len() - self.num_samples..].to_vec()
        }

        fn num_warmups(&self) -> usize {
            self.executed.borrow().len() - self.num_samples
        }
    }

    impl Benchmark for TestBenchmark {
        type Input = usize;
        type Output = ();

        fn prepare(&self) -> Self::Input {
            let mut prepared = self.prepared.borrow_mut();
            let id = *prepared;
            *prepared += 1;
            id
        }

        fn execute(&self, input: Self::Input) -> Self::Output {
            std::thread::sleep(self.execution_time);
            self.executed.borrow_mut().push(input);
        }

        fn name(&self) -> String {
            "test".into()
        }

        fn sync(&self) {}

        fn warmup(&self) -> Duration {
            self.warmup
        }

        fn num_inputs(&self) -> usize {
            self.num_inputs
        }

        fn num_samples(&self) -> usize {
            self.num_samples
        }

        fn prepare_cloned(&self) -> bool {
            self.prepare_cloned
        }
    }

    /// Keeps the default [Benchmark::warmup()] implementation, unlike [TestBenchmark].
    struct DefaultBenchmark;

    impl Benchmark for DefaultBenchmark {
        type Input = ();
        type Output = ();

        fn prepare(&self) -> Self::Input {}
        fn execute(&self, _input: Self::Input) -> Self::Output {}
        fn name(&self) -> String {
            "default".into()
        }
        fn sync(&self) {}
    }

    #[test]
    #[serial_test::serial(bench_env)]
    fn warmup_defaults_to_200ms() {
        unsafe { std::env::remove_var("BENCH_WARMUP_MS") };

        assert_eq!(DefaultBenchmark.warmup(), Duration::from_millis(200));
        assert_eq!(DefaultBenchmark.num_inputs(), 1);
    }

    #[test]
    #[serial_test::serial(bench_env)]
    fn warmup_can_be_overridden_by_the_environment() {
        unsafe { std::env::set_var("BENCH_WARMUP_MS", "42") };
        let warmup = DefaultBenchmark.warmup();

        // Invalid values fall back to the default.
        unsafe { std::env::set_var("BENCH_WARMUP_MS", "not-a-number") };
        let invalid = DefaultBenchmark.warmup();
        unsafe { std::env::remove_var("BENCH_WARMUP_MS") };

        assert_eq!(warmup, Duration::from_millis(42));
        assert_eq!(invalid, Duration::from_millis(200));
    }

    #[test]
    fn warmup_scales_with_its_duration_not_with_a_count() {
        let bench = TestBenchmark {
            warmup: Duration::from_millis(50),
            execution_time: Duration::from_millis(1),
            ..Default::default()
        };
        bench.run(TimingMethod::System);

        // Roughly 50 executions of 1ms, way more than the 5 a fixed count would have given.
        assert!(
            bench.num_warmups() > 10,
            "expected many warmup executions, got {}",
            bench.num_warmups()
        );
    }

    #[test]
    fn warmup_always_runs_at_least_once() {
        let bench = TestBenchmark::default();
        bench.run(TimingMethod::System);

        assert_eq!(bench.num_warmups(), 1);
    }

    #[test]
    fn single_input_is_reused_by_default() {
        let bench = TestBenchmark {
            num_samples: 4,
            ..Default::default()
        };
        let durations = bench.run(TimingMethod::System);

        assert_eq!(*bench.prepared.borrow(), 1);
        assert_eq!(bench.sampled(), vec![0, 0, 0, 0]);
        assert_eq!(durations.durations.len(), 4);
    }

    #[test]
    fn distinct_inputs_are_cycled_through() {
        let bench = TestBenchmark {
            num_inputs: 3,
            num_samples: 7,
            ..Default::default()
        };
        bench.run(TimingMethod::System);

        // One prepare per input, and every execution cycles over them.
        assert_eq!(*bench.prepared.borrow(), 3);
        assert_eq!(bench.sampled(), vec![0, 1, 2, 0, 1, 2, 0]);
    }

    #[test]
    fn zero_inputs_is_treated_as_one() {
        let bench = TestBenchmark {
            num_inputs: 0,
            ..Default::default()
        };
        bench.run(TimingMethod::System);

        assert_eq!(*bench.prepared.borrow(), 1);
    }

    #[test]
    fn uncloned_inputs_are_prepared_for_every_execution() {
        let bench = TestBenchmark {
            prepare_cloned: false,
            num_inputs: 3,
            num_samples: 3,
            ..Default::default()
        };
        bench.run(TimingMethod::System);

        // `num_inputs` is irrelevant here: every execution gets its own fresh input.
        let executed = bench.executed.borrow().clone();
        assert_eq!(*bench.prepared.borrow(), executed.len());
        assert_eq!(bench.sampled(), vec![1, 2, 3]);
    }
}
