# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Windows port of .ci/build-shim.sh / .ci/common-build.sh's common_build():
# a fast "does the C++ shim compile" build -- no GPU kernel images
# (AOTRITON_NOIMAGE_MODE), no python bindings (AOTRITON_NO_PYTHON). Two
# things dropped relative to the Linux script, both Linux/GNU-only and not
# applicable here: the `-fuse-ld=mold` linker flags, and any python/venv
# bootstrapping (CMakeLists.txt still needs a Python3 interpreter for
# codegen -- see find_package(Python3 ... REQUIRED) -- but AOTRITON_NO_PYTHON
# means this build never touches torch/pyaotriton, so no venv/pip setup is
# needed from this script itself).
#
# Run from "Developer PowerShell for VS 2022" (need cl.exe/link.exe on PATH).
#
# Usage:
#   .\build-shim.ps1 <target arch>[;<arch>...]
#
# Overrides (matching common-build.sh's env-var overrides):
#   $env:AOTRITON_BUILD_PATH          - explicit build dir (default: build-<major>.<minor>-shim-<arch>)
#   $env:AOTRITON_INSTALL_PATH        - install prefix (default: ./install_dir)
#   $env:AOTRITON_NAME_SUFFIX_OVERRIDE - AOTRITON_NAME_SUFFIX (default: 123)

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$TargetArch
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SourceDir = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$RootCMakeLists = Join-Path $SourceDir "CMakeLists.txt"

# --- ROCM_PATH sanity check -------------------------------------------------
# CMakeLists.txt only warns and falls back to /opt/rocm (a Linux path) when
# ROCM_PATH is unset -- on Windows that fallback is never valid, so fail
# fast here instead of letting find_package(hip) produce a generic error.
if (-not $env:ROCM_PATH) {
    # Same fallback CMakeLists.txt's warning suggests on Linux (`rocm-sdk path
    # --root`): theRock's Python package reports its own install root, so use
    # it when present instead of requiring the caller to set ROCM_PATH by hand.
    $rocmSdkCmd = Get-Command rocm-sdk -ErrorAction SilentlyContinue
    if ($rocmSdkCmd) {
        $resolved = & rocm-sdk path --root 2>$null
        if ($LASTEXITCODE -eq 0 -and $resolved) {
            $env:ROCM_PATH = $resolved.Trim()
            Write-Host "ROCM_PATH not set; using 'rocm-sdk path --root': $env:ROCM_PATH"
        }
    }
}
if (-not $env:ROCM_PATH) {
    throw "ROCM_PATH is not set, and 'rocm-sdk path --root' is unavailable. " +
          "Set ROCM_PATH to your HIP SDK install root " +
          "(the directory containing lib\, bin\, include\)."
}
$HipConfig = Join-Path $env:ROCM_PATH "lib\cmake\hip\hip-config.cmake"
if (-not (Test-Path $HipConfig)) {
    throw "hip-config.cmake not found at $HipConfig -- ROCM_PATH must be " +
          "the HIP SDK ROOT, not a subdirectory. Current ROCM_PATH: $env:ROCM_PATH"
}
$env:PATH = "$env:ROCM_PATH\bin;$env:ROCM_PATH\llvm\bin;$env:PATH"

# --- Version numbers, mirroring common-vars.sh's grep-based parsing --------
$cmakeListsText = Get-Content $RootCMakeLists -Raw
$aotritonMajor = [regex]::Match($cmakeListsText, 'set\(AOTRITON_VERSION_MAJOR_INT\s+(\d+)\)').Groups[1].Value
$aotritonMinor = [regex]::Match($cmakeListsText, 'set\(AOTRITON_VERSION_MINOR_INT\s+(\d+)\)').Groups[1].Value
if (-not $aotritonMajor -or -not $aotritonMinor) {
    throw "Could not parse AOTRITON_VERSION_{MAJOR,MINOR}_INT from $RootCMakeLists"
}

$Suffix = if ($env:AOTRITON_NAME_SUFFIX_OVERRIDE) { $env:AOTRITON_NAME_SUFFIX_OVERRIDE } else { "123" }
$ArchForDirName = $TargetArch -replace ';', '_'

$BuildDir = if ($env:AOTRITON_BUILD_PATH) {
    $env:AOTRITON_BUILD_PATH
} else {
    Join-Path $SourceDir "build-$aotritonMajor.$aotritonMinor-shim-$ArchForDirName"
}
$InstallPrefix = if ($env:AOTRITON_INSTALL_PATH) { $env:AOTRITON_INSTALL_PATH } else { "./install_dir" }

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
Push-Location $BuildDir
try {
    cmake $SourceDir `
        -DCMAKE_INSTALL_PREFIX="$InstallPrefix" `
        -DCMAKE_BUILD_TYPE=Release `
        -DAOTRITON_TARGET_ARCH="$TargetArch" `
        -DAOTRITON_NAME_SUFFIX="$Suffix" `
        -DAOTRITON_NOIMAGE_MODE=ON `
        -DAOTRITON_GPU_BUILD_TIMEOUT=0 `
        -DAOTRITON_NO_PYTHON=ON `
        -G Ninja
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed with exit code $LASTEXITCODE" }

    ninja install/strip
    if ($LASTEXITCODE -ne 0) { throw "ninja install/strip failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
