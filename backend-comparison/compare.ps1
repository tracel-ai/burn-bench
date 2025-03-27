<#
.SYNOPSIS
    Runs Burn benchmarks for multiple versions or git commits
.DESCRIPTION
    This script allows running burnbench for different Burn versions,
    updating Cargo.toml dynamically and logging benchmarks results.
.EXAMPLE
    .\burnbench.ps1 0.16.1 --benches unary --backends ndarray
#>

param(
    [Parameter(Position=0, ValueFromRemainingArguments=$true)]
    [string[]]$InputArgs
)

# Fail on first error
$ErrorActionPreference = 'Stop'

$CARGO_TOML = "Cargo.toml"
$BACKUP_FILE = "$CARGO_TOML.bak"
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOGFILE = "burnbench_$TIMESTAMP.log"

function Show-Usage {
    Write-Host "Usage: .\burnbench.ps1 <version1> [version2...] <burnbench_args>"
    Write-Host ""
    Write-Host "Arguments:"
    Write-Host "  <version>         One or more Burn version, git branch or commit hash"
    Write-Host "  <burnbench_args>  Any argument(s) to pass to burnbench"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\burnbench.ps1 0.16.1 --benches unary --backends ndarray"
    exit 1
}

function Is-Version {
    param([string]$Version)
    return $Version -match '^\d+\.\d+(\.\d+)?(-[a-zA-Z0-9]+)?$'
}

function VersionLessThan017 {
    param([string]$Version)
    $parts = $Version -split '\.'
    return ($parts[0] -eq '0' -and [int]$parts[1] -lt 17)
}

function Replace-FeatureFlagsLessThan017 {
    param([string]$CargoToml)
    
    (Get-Content $CargoToml) | ForEach-Object {
        $_ -replace 'cuda = \["burn/cuda"\]', 'cuda = ["burn/cuda-jit"]' `
           -replace 'hip = \["burn/hip"\]', 'hip = ["burn/hip-jit"]' `
           -replace 'wgpu-spirv = \["burn/vulkan", "burn/autotune"\]', 'wgpu-spirv = ["burn/wgpu-spirv", "burn/autotune"]' `
           -replace 'ndarray-simd = \["burn/ndarray", "burn/simd"\]', 'ndarray-simd = ["burn/ndarray"]'
    } | Set-Content $CargoToml
}

function Replace-FeatureFlagsGreaterEqual017 {
    param([string]$CargoToml)
    
    (Get-Content $CargoToml) | ForEach-Object {
        $_ -replace 'cuda = \["burn/cuda-jit"\]', 'cuda = ["burn/cuda"]' `
           -replace 'hip = \["burn/hip-jit"\]', 'hip = ["burn/hip"]' `
           -replace 'wgpu-spirv = \["burn/wgpu-spirv", "burn/autotune"\]', 'wgpu-spirv = ["burn/vulkan", "burn/autotune"]' `
           -replace 'ndarray-simd = \["burn/ndarray"\]', 'ndarray-simd = ["burn/ndarray", "burn/simd"]'
    } | Set-Content $CargoToml
}

function Update-CargoToml {
    param(
        [string]$Version,
        [string]$CargoToml
    )

    if (Is-Version $Version) {
        Write-Host "Applying Burn version: $Version"

        # For version, update both burn and burn-common
        (Get-Content $CargoToml) | ForEach-Object {
            $_ -replace 'burn = \{ .+, default-features = false \}', "burn = { version = ""$Version"", default-features = false }" `
               -replace 'burn-common = \{ .+ \}', "burn-common = { version = ""$Version"" }"
        } | Set-Content $CargoToml

        # Handle feature flags for previous versions
        if (VersionLessThan017 $Version) {
            Write-Host "Version < 0.17.0 detected, changing feature flags"
            Replace-FeatureFlagsLessThan017 $CargoToml
            # Pin bincode pre-release (used in burn < 0.17)
            (Get-Content $CargoToml) -replace '(\[dependencies\])', "`$1`r`nbincode = `"=2.0.0-rc.3`"`r`nbincode_derive = `"=2.0.0-rc.3`"" | Set-Content $CargoToml
        }
        else {
            Write-Host "Version >= 0.17.0 detected, using cuda, hip, vulkan and simd feature flags"
            Replace-FeatureFlagsGreaterEqual017 $CargoToml
        }
    }
    else {
        # Support commit hashes (7 to 40 hexadecimal characters) or branch names
        $GitRev = ""
        if ($Version -match "^[0-9a-f]{7,40}$") {
            $GitRev = "rev = `"$Version`""
        } else {
            $GitRev = "branch = `"$Version`""
        }
        Write-Host "Applying Burn git commit: $GitRev"

        # For git commit, update both burn and burn-common
        (Get-Content $CargoToml) | ForEach-Object {
            $_ -replace 'burn = \{ .+, default-features = false \}', "burn = { git = ""https://github.com/tracel-ai/burn"", $GitRev, default-features = false }" `
               -replace 'burn-common = \{ .+ \}', "burn-common = { git = ""https://github.com/tracel-ai/burn"", $GitRev }"
        } | Set-Content $CargoToml

        Write-Host "Warning: Assuming version >= 0.17 for git commit, you may need to manually check the cuda feature flag name."
        Replace-FeatureFlagsGreaterEqual017 $CargoToml
    }
}

function Run-Burnbench {
    param(
        [string]$Version,
        [string[]]$RunArgs
    )

    Write-Host "----------------------------------------------"
    Write-Host "Running burnbench for version: $Version"
    $Command = "cargo run --release --bin burnbench -- run $RunArgs"
    Write-Host "Command: $Command"
    Write-Host "----------------------------------------------"

    # Run command and capture output
    Invoke-Expression $Command | Tee-Object -Append -FilePath $LOGFILE

    Write-Host "----------------------------------------------"
    Write-Host "Completed burnbench run for version: $Version"
    Write-Host "----------------------------------------------"
    Write-Host ""
}

# Check for help
if ($InputArgs -contains '-h' -or $InputArgs -contains '--help') {
    Show-Usage
}

# Parse command-line arguments
$Versions = @()
$BurnbenchArgs = @()
$ParseBurnbenchArgs = $false

foreach ($arg in $InputArgs) {
    if ($arg.StartsWith("-") -and -not $ParseBurnbenchArgs) {
        # Any argument that starts with `-` will be interpreted as a burnbench arg
        # All arguments, short or long, require this prefix (-b, --benches, -B, --backends, etc.)
        $ParseBurnbenchArgs = $true
        $BurnbenchArgs += $arg
    }
    elseif ($ParseBurnbenchArgs) {
        $BurnbenchArgs += $arg
    }
    else {
        $Versions += $arg
    }
}

if ($Versions.Count -eq 0) {
    Write-Host "Error: No Burn versions or commits specified."
    Show-Usage
}

if ($BurnbenchArgs.Count -eq 0) {
    Write-Host "Error: No burnbench arguments provided."
    Show-Usage
}

# Summary
Write-Host "========================================================"
Write-Host "           BURN BENCHMARK EXECUTION SUMMARY             "
Write-Host "========================================================"
Write-Host "Versions to benchmark:"
for ($i = 0; $i -lt $Versions.Count; $i++) {
    Write-Host "  $($i + 1). $($Versions[$i])"
}
Write-Host ""

if ($BurnbenchArgs.Count -gt 0) {
    Write-Host "Burnbench arguments: $($BurnbenchArgs -join ' ')"
    Write-Host ""
    Write-Host "The following command will be executed:"
    Write-Host "  cargo run --release --bin burnbench -- run $($BurnbenchArgs -join ' ')"
}
Write-Host "========================================================"
Write-Host ""

# Confirmation prompt
$confirmation = Read-Host "Do you want to proceed? ([y]/n)"
if ($confirmation -ne 'y' -and $confirmation -ne '') {
    Write-Host "Operation cancelled."
    exit 0
}

# Backup original Cargo.toml
Copy-Item $CARGO_TOML $BACKUP_FILE
Write-Host "Created backup of Cargo.toml at $BACKUP_FILE"

# Trap to restore backup on script exit
$scriptErrorHandler = {
    # Restore the original Cargo.toml
    if (Test-Path $BACKUP_FILE) {
        Copy-Item $BACKUP_FILE $CARGO_TOML
    }
}

try {
    # Process each version
    foreach ($Version in $Versions) {
        # Update Cargo.toml for this version
        Update-CargoToml $Version $CARGO_TOML
        Write-Host "Cargo.toml updated successfully with version: $Version"

        # Build and run burnbench with provided arguments
        Run-Burnbench $Version $BurnbenchArgs

    # Restore the original Cargo.toml to avoid leaving it in an unexpected state
        Copy-Item $BACKUP_FILE $CARGO_TOML
    }
}
catch {
    Write-Host "An error occurred: $_"
    $scriptErrorHandler.Invoke()
}
finally {
    # Clean up backup file
    if (Test-Path $BACKUP_FILE) {
        Remove-Item $BACKUP_FILE
    }
}

Write-Host ""
Write-Host "All benchmark runs completed!"
Write-Host "Check out the aggregated results: $LOGFILE"