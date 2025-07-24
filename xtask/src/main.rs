use tracel_xtask::prelude::*;

#[macros::base_commands(Build, Bump, Check, Compile, Fix, Test)]
enum Command {}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = init_xtask::<Command>(parse_args::<Command>()?)?;
    dispatch_base_commands(args)?;
    Ok(())
}
