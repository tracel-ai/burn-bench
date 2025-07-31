pub mod __private;
mod benchmark;
mod persistence;
mod runner;

pub(crate) mod system_info;

pub use benchmark::*;
pub use persistence::*;
pub use runner::*;
pub use system_info::*;

const BENCHMARKS_TARGET_DIR: &str = "target/benchmarks";
const TRACEL_CI_SERVER_BASE_URL: &str = if cfg!(debug_assertions) {
    // development
    "http://localhost:8000/v1/"
} else {
    // production
    "https://user-benchmark-server-812794505978.northamerica-northeast1.run.app/v1/"
};

const BENCHMARK_WEBSITE_URL: &str = if cfg!(debug_assertions) || cfg!(test) {
    "http://localhost:4321/"
} else {
    "https://burn.dev/"
};

#[macro_export]
macro_rules! ci_errorln {
    ($($arg:tt)*) => {{
        if std::env::var("CI").is_ok() {
            // Print to stdout with ::error prefix for GitHub Actions
            println!("::error ::{}", format!($($arg)*));
        } else {
            // Local dev: print to stderr as usual
            eprintln!($($arg)*);
        }
    }};
}
