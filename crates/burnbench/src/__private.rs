use tracing_subscriber::{
    Layer,
    filter::{LevelFilter, filter_fn},
    layer::SubscriberExt,
    registry,
    util::SubscriberInitExt,
};

/// Simple parse to retrieve additional argument passed to cargo bench command
/// We cannot use clap here as clap parser does not allow to have unknown arguments.
pub fn get_argument<'a>(args: &'a [String], arg_name: &'a str) -> Option<&'a str> {
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            arg if arg == arg_name && i + 1 < args.len() => {
                return Some(&args[i + 1]);
            }
            _ => i += 1,
        }
    }
    None
}

/// Specialized function to retrieve the sharing token
pub fn get_sharing_token(args: &[String]) -> Option<&str> {
    get_argument(args, "--sharing-token")
}

/// Specialized function to retrieve the sharing URL
pub fn get_sharing_url(args: &[String]) -> Option<&str> {
    get_argument(args, "--sharing-url")
}

pub fn init_log() -> Result<(), String> {
    let layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_filter(LevelFilter::INFO)
        .with_filter(filter_fn(|m| {
            if let Some(path) = m.module_path()
                && path.starts_with("wgpu")
            {
                return false;
            }
            true
        }));

    let result = registry().with(layer).try_init();

    if result.is_err() {
        update_panic_hook();
    }

    result.map_err(|err| format!("{err:?}"))
}

fn update_panic_hook() {
    let hook = std::panic::take_hook();

    std::panic::set_hook(Box::new(move |info| {
        log::error!("PANIC => {}", info);
        hook(info);
    }));
}
