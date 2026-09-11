# One-click installer: venv + package + ONNX model + Cursor MCP config.
# Usage:
#   .\install.ps1
#   irm https://ghfast.top/https://raw.githubusercontent.com/HG-ha/depth-anything-mcp/main/install.ps1 | iex

param(
    [string]$Repo = $(if ($env:DEPTH_ANYTHING_REPO) { $env:DEPTH_ANYTHING_REPO } else { "https://github.com/HG-ha/depth-anything-mcp.git" }),
    [ValidateSet("dynamic", "quantized")]
    [string]$Variant = "dynamic"
)

function Use-CnMirror {
    $value = if ($null -ne $env:DEPTH_ANYTHING_MIRROR) { $env:DEPTH_ANYTHING_MIRROR } else { "cn" }
    return @("0", "off", "false", "no", "none", "official") -notcontains $value.ToLower()
}

function Mirror-GitHubUrl([string]$Url) {
    if (-not (Use-CnMirror)) { return $Url }
    if ($Url -notmatch "github.com" -and $Url -notmatch "githubusercontent.com") { return $Url }
    if ($Url -match "ghfast.top|gh-proxy.com|ghproxy|gitclone.com") { return $Url }
    $proxy = if ($env:DEPTH_ANYTHING_GITHUB_PROXY) { $env:DEPTH_ANYTHING_GITHUB_PROXY.TrimEnd("/") } else { "https://ghfast.top" }
    return "$proxy/$Url"
}

if (-not $env:DEPTH_ANYTHING_REPO) {
    $Repo = Mirror-GitHubUrl $Repo
}

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
$PipIndex = @()
if (Use-CnMirror) {
    $index = if ($env:DEPTH_ANYTHING_PYPI_INDEX) { $env:DEPTH_ANYTHING_PYPI_INDEX } else { "https://pypi.tuna.tsinghua.edu.cn/simple" }
    $PipIndex = @("-i", $index, "--trusted-host", "pypi.tuna.tsinghua.edu.cn")
}
& $VenvPython -m pip install -U pip @PipIndex
& $VenvPython -m pip install @PipIndex -e $installSpec
& $VenvPython -m depth_anything_mcp.install --home $HomeDir --python $VenvPython --variant $Variant

Write-Host ""
Write-Host "Done. Reload MCP in Cursor, then try estimate_image_depth."
