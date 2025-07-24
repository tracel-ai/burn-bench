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
const USER_BENCHMARK_SERVER_URL: &str = if cfg!(debug_assertions) {
    // development
    "http://localhost:8000/v1/"
} else {
    // production
    "https://user-benchmark-server-812794505978.northamerica-northeast1.run.app/v1/"
};

const USER_BENCHMARK_WEBSITE_URL: &str = if cfg!(debug_assertions) || cfg!(test) {
    "http://localhost:4321/"
} else {
    "https://burn.devel/"
};
