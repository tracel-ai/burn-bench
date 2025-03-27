#!/bin/bash

set -e

CARGO_TOML="Cargo.toml"
BACKUP_FILE="${CARGO_TOML}.bak"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="burnbench_${TIMESTAMP}.log"

restore_backup() {
    if [[ -f "$BACKUP_FILE" ]]; then
        cp "$BACKUP_FILE" "$CARGO_TOML"
    fi
}

# Ensure the backup is restored on exit
trap restore_backup EXIT

show_usage() {
    echo "Usage: $0 <version1> [version2...] <burnbench_args>"
    echo ""
    echo "Arguments:"
    echo "  <version>         One or more Burn versions or git commits"
    echo "  <burnbench_args>  Any argument(s) to pass to burnbench"
    echo ""
    echo "Examples:"
    echo "  $0 0.16.1 --benches unary --backends ndarray"
    exit 1
}

is_version() {
    [[ "$1" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?(-[a-zA-Z0-9]+)?$ ]]
}

version_lt_0_17() {
    # Extract major and minor components
    local major=$(echo "$1" | cut -d. -f1)
    local minor=$(echo "$1" | cut -d. -f2)

    # Check if major is 0 and minor is less than 17
    [[ "$major" -eq 0 && "$minor" -lt 17 ]]
}

replace_feature_flags_lt_0_17() {
    local CARGO_TOML="$1"
    sed -i.tmp -E \
            -e 's/cuda = \["burn\/cuda"\]/cuda = \["burn\/cuda-jit"\]/g' \
            -e 's/hip = \["burn\/hip"\]/hip = \["burn\/hip-jit"\]/g' \
            -e 's/wgpu-spirv = \["burn\/vulkan", "burn\/autotune"\]/wgpu-spirv = \["burn\/wgpu-spirv", "burn\/autotune"\]/g' \
            -e 's/ndarray-simd = \["burn\/ndarray", "burn\/simd"\]/ndarray-simd = \["burn\/ndarray"\]/g' \
            "$CARGO_TOML"
}

replace_feature_flags_ge_0_17() {
    local CARGO_TOML="$1"
    sed -i.tmp -E \
        -e 's/cuda = \["burn\/cuda-jit"\]/cuda = \["burn\/cuda"\]/g' \
        -e 's/hip = \["burn\/hip-jit"\]/hip = \["burn\/hip"\]/g' \
        -e 's/wgpu-spirv = \["burn\/wgpu-spirv", "burn\/autotune"\]/wgpu-spirv = \["burn\/vulkan", "burn\/autotune"\]/g' \
        -e 's/ndarray-simd = \["burn\/ndarray"\]/ndarray-simd = \["burn\/ndarray", "burn\/simd"\]/g' \
        "$CARGO_TOML"
}

update_cargo_toml() {
    local VERSION="$1"
    local CARGO_TOML="$2"

    if is_version "$VERSION"; then
        echo "Applying Burn version: $VERSION"

        # For version, update both burn and burn-common
        sed -i.tmp -E \
            -e "s|burn = \{ .+, default-features = false \}|burn = { version = \"$VERSION\", default-features = false }|g" \
            -e "s|burn-common = \{ .+ \}|burn-common = { version = \"$VERSION\" }|g" \
            "$CARGO_TOML"

        # Handle feature flags for previous versions
        if version_lt_0_17 "$VERSION"; then
            echo "Version < 0.17.0 detected, changing feature flags"
            replace_feature_flags_lt_0_17 "$CARGO_TOML"
            # Pin bincode pre-release (used in burn < 0.17)
            sed -i '/^\[dependencies\]/a bincode = "=2.0.0-rc.3"\nbincode_derive = "=2.0.0-rc.3"' Cargo.toml
        else
            echo "Version >= 0.17.0 detected, using cuda, hip, vulkan and simd feature flags"
            replace_feature_flags_ge_0_17 "$CARGO_TOML"
        fi
    else
        echo "Applying Burn git commit: $VERSION"

        # For git commit, update both burn and burn-common
        sed -i.tmp -E \
            -e "s|burn = \{ .+, default-features = false \}|burn = { git = \"https://github.com/tracel-ai/burn\", rev = \"$VERSION\", default-features = false }|g" \
            -e "s|burn-common = \{ .+ \}|burn-common = { git = \"https://github.com/tracel-ai/burn\", rev = \"$VERSION\" }|g" \
            "$CARGO_TOML"

        echo "Warning: Assuming version >= 0.17 for git commit, you may need to manually check the cuda feature flag name."
        replace_feature_flags_ge_0_17 "$CARGO_TOML"
    fi

    # Clean up temporary files created by -i.tmp
    rm -f "${CARGO_TOML}.tmp"
}

run_burnbench() {
    local VERSION="$1"
    shift
    local ARGS=("$@")

    echo "----------------------------------------------"
    echo "Running burnbench for version: $VERSION"
    echo "Command: cargo run --release --bin burnbench -- run ${ARGS[*]}"
    echo "----------------------------------------------"

    cargo run --release --color=always --bin burnbench -- run "${ARGS[@]}" | tee -a ${LOGFILE}

    echo "----------------------------------------------"
    echo "Completed burnbench run for version: $VERSION"
    echo "----------------------------------------------"
    echo ""
}

# Check for --help
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_usage
fi

# Parse command-line arguments
VERSIONS=()
BURNBENCH_ARGS=()
PARSE_BURNBENCH_ARGS=false

for arg in "$@"; do
    # Keep support for `--` separator as before, but not required anymore to match powershell usage (special token)
    if [[ "$arg" == "--" && $PARSE_BURNBENCH_ARGS == false ]]; then
        PARSE_BURNBENCH_ARGS=true
    elif [[ "$arg" == -* && $PARSE_BURNBENCH_ARGS == false ]]; then
        # Any argument that starts with `-` will be interpreted as a burnbench arg
        # All arguments, short or long, require this prefix (-b, --benches, -B, --backends, etc.)
        PARSE_BURNBENCH_ARGS=true
        BURNBENCH_ARGS+=("$arg")
    elif [[ $PARSE_BURNBENCH_ARGS == true ]]; then
        BURNBENCH_ARGS+=("$arg")
    else
        VERSIONS+=("$arg")
    fi
done

if [ ${#VERSIONS[@]} == 0  ]; then
    echo "Error: No Burn versions or commits specified."
    echo ""
    show_usage
fi

if [ ${#BURNBENCH_ARGS[@]} == 0 ]; then
    echo "Error: No burnbench arguments provided."
    echo ""
    show_usage
fi

# Summary
echo "========================================================"
echo "           BURN BENCHMARK EXECUTION SUMMARY             "
echo "========================================================"
echo "Versions to benchmark:"
for ((i=0; i<${#VERSIONS[@]}; i++)); do
    echo "  $(($i+1)). ${VERSIONS[$i]}"
done
echo ""

if [ ${#BURNBENCH_ARGS[@]} -gt 0 ]; then
    echo "Burnbench arguments: ${BURNBENCH_ARGS[*]}"
    echo ""
    echo "The following command will be executed:"
    echo "  cargo run --release --bin burnbench -- run ${BURNBENCH_ARGS[*]}"
fi
echo "========================================================"
echo ""

read -p "Do you want to proceed? ([y]/n): " -n 1 -r
echo ""
if [[ -n $REPLY && ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 0
fi

# Backup the original Cargo.toml
cp "$CARGO_TOML" "$BACKUP_FILE"
echo "Created backup of Cargo.toml at ${BACKUP_FILE}"

# Process each version
for VERSION in "${VERSIONS[@]}"; do
    # Update Cargo.toml for this version
    update_cargo_toml "$VERSION" "$CARGO_TOML"
    echo "Cargo.toml updated successfully with version: $VERSION"

    # Build and run burnbench with provided arguments
    run_burnbench "$VERSION" "${BURNBENCH_ARGS[@]}"

    # Restore the original Cargo.toml to avoid leaving it in an unexpected state
    restore_backup
done

# Remove the trap
trap - EXIT

echo ""
echo "All benchmark runs completed!"
echo "Check out the aggregated results: $LOGFILE"
