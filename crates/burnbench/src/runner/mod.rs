pub(crate) mod auth;
mod base;
mod dependency;
mod processor;
mod progressbar;
mod reports;
mod workflow;

pub use base::*;

#[macro_export]
macro_rules! group {
    // group!()
    ($($arg:tt)*) => {
        let title = format!($($arg)*);
        if std::env::var("CI").is_ok() {
            println!("::group::{title}")
        } else {
            println!("{title}")
        }
    };
}

#[macro_export]
macro_rules! endgroup {
    // endgroup!()
    () => {
        if std::env::var("CI").is_ok() {
            println!("::endgroup::")
        }
    };
}
