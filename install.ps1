# One-click installer: venv + package + ONNX model + Cursor MCP config.
# Usage:
#   .\install.ps1
#   irm https://raw.githubusercontent.com/HG-ha/depth-anything-mcp/main/install.ps1 | iex

param(
    [string]$Repo = $(if ($env:DEPTH_ANYTHING_REPO) { $env:DEPTH_ANYTHING_REPO } else { "https://github.com/HG-ha/depth-anything-mcp.git" }),
    [ValidateSet("dynamic", "quantized")]
    [string]$Variant = "dynamic"
)

$ErrorActionPreference = "Stop"
$HomeDir = Join-Path $HOME ".depth-anything-mcp"
$VenvDir = Join-Path $HomeDir "venv"
$SrcDir = Join-Path $HomeDir "src"

function Find-Python {
    $candidates = @(
        @{ File = "py"; Args = @("-3") },
        @{ File = "python"; Args = @() },
        @{ File = "python3"; Args = @() }
    )
    foreach ($item in $candidates) {
        $cmd = Get-Command $item.File -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        try {
            $ver = & $cmd.Source @($item.Args + @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"))
            $parts = $ver.Trim().Split(".")
            if ([int]$parts[0] -gt 3 -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge 10)) {
                return @{ Exe = $cmd.Source; Args = $item.Args }
            }
        } catch {
            continue
        }
    }
    throw "Python 3.10+ is required."
}

$py = Find-Python
New-Item -ItemType Directory -Force -Path $HomeDir | Out-Null

$localRoot = $null
if ($PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot "pyproject.toml"))) {
    $localRoot = $PSScriptRoot
}

if ($localRoot) {
    Write-Host "Installing from local checkout: $localRoot"
    $installSpec = $localRoot
} else {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "git is required to clone $Repo"
    }
    if (Test-Path (Join-Path $SrcDir "pyproject.toml")) {
        Write-Host "Updating $SrcDir"
        git -C $SrcDir pull --ff-only
    } else {
        Write-Host "Cloning $Repo"
        if (Test-Path $SrcDir) { Remove-Item -Recurse -Force $SrcDir }
        git clone --depth 1 $Repo $SrcDir
    }
    $installSpec = $SrcDir
}

if (-not (Test-Path (Join-Path $VenvDir "Scripts\python.exe"))) {
    Write-Host "Creating venv $VenvDir"
    & $py.Exe @($py.Args + @("-m", "venv", $VenvDir))
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
& $VenvPython -m pip install -U pip
& $VenvPython -m pip install -e $installSpec
& $VenvPython -m depth_anything_mcp.install --home $HomeDir --python $VenvPython --variant $Variant

Write-Host ""
Write-Host "Done. Reload MCP in Cursor, then try estimate_image_depth."
