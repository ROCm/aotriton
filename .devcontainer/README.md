# AOTriton gfx908 development in Coder

This environment runs as root with the checkout and Codex state on the
`/workspaces` PVC. It uses the digest-pinned ROCm 10.1.0 / Ubuntu 26.04 / PyTorch
2.14.0 / Python 3.14 image and adds AOTriton's CMake, Ninja, ccache, Python
headers, and compression/compiler build dependencies.

PyTorch supplies tensor allocation and test references. The local AOTriton
library is built with Python bindings and the namespace/library suffix `123`,
so the tests load it alongside the AOTriton shipped with PyTorch. Building this
checkout does not replace PyTorch's built-in SDPA library.

## Coder template

Use the existing Coder/Envbuilder template with the AOTriton fork as its Git
repository and `.devcontainer/devcontainer.json` as its devcontainer config.
The [Pod integration fragment](coder-pod.example.yaml) uses
`ENVBUILDER_WORKSPACE_FOLDER=/workspaces/aotriton`, reserves one `amd.com/gpu`,
selects the MI100 node, mounts the workspace PVC, and provides an 8 GiB
RAM-backed `/dev/shm`. Merge it into the existing template; it is not a
standalone manifest. The referenced Terraform template from the AITER README
was not included in this copy.

GPU access comes from the Kubernetes AMD device plugin. The devcontainer does
not need nested Docker/Podman or install a host AMDGPU driver. Coder's agent,
token, Envbuilder image, Git settings, and cache stay in the existing template.

```text
/workspaces/                  <- persistent PVC
  .codex/                     <- Codex configuration, auth and sessions
  aotriton/                   <- fork checkout
    build-compilers/          <- LLVM tarball and FlyDSL wheel cache
    build-gfx908/             <- CMake build, compiler venv and local install
```

Envbuilder's workspace path must point to the checkout root. Adjust the Pod's
workspace setting if the repository basename/path differs. Root sessions need
write access to the PVC. The `/dev/shm` volume is temporary, and its usage
counts against the Pod's memory limit.

## Bootstrap

The post-create hook installs an optional organization CA from
`CUSTOM_CA_CERT_B64`, performs static tooling/storage checks, initializes
submodules, installs development and tuning dependencies, and installs the
`aotriton` code generator and `aotriton.tune` in editable mode. The development
interpreter is `/opt/aotriton-venv/bin/python`; the venv is rebuilt with the
image rather than persisted across Python/ROCm changes.

Bootstrap does not compile kernels or execute GPU checks. In the Coder terminal:

```bash
cd /workspaces/aotriton
python .devcontainer/doctor.py
codex login --device-auth
```

The doctor checks tools, PVC/Codex storage, gfx908 assembly, GPU visibility,
and FP32/FP16/BF16 GEMM on the exposed MI100. Use `--static` to skip GPU access.
Set `AOTRITON_DEV_INSTALL=0` in `containerEnv` to skip dependency/submodule
setup while troubleshooting, then rerun `bash .devcontainer/bootstrap.sh`
when ready.

## Build gfx908 kernels and bindings

The checkout currently pins a patched LLVM for FlyDSL. Its CMake build requires
a FlyDSL wheel built against that LLVM for every full image build, including
gfx908. Prepare it explicitly inside the workspace:

```bash
cd /workspaces/aotriton
AOTRITON_FLYDSL_WHEEL="$(bash .devcontainer/prepare-flydsl.sh)"
export AOTRITON_FLYDSL_WHEEL
bash .devcontainer/build.sh
```

Stop if compiler preparation fails. It builds the pinned LLVM and FlyDSL using
the repository's native `.ci/runc-build-*.sh` helpers and FlyDSL patches, with
no container daemon. It can take substantial time and disk space on the first
run; subsequent runs reuse the LLVM tarball and matching wheel. Source/build
trees are retained under `build-compilers` for troubleshooting. The LLVM build
defaults to two jobs (`AOTRITON_COMPILER_JOBS`); FlyDSL's own build script also
controls its parallelism. `LLVM_TARBALL_DISTRO=ubuntu26.04` keeps this image's
compiler cache separate from the previous Ubuntu base.

An existing compatible wheel can be supplied instead:

```bash
export AOTRITON_FLYDSL_WHEEL=/absolute/path/to/flydsl-cp314-wheel.whl
bash .devcontainer/build.sh
```

The build helper targets `gfx908`, retains debug symbols, builds actual kernel
images (`AOTRITON_NOIMAGE_MODE=OFF`) and `pyaotriton`, and installs under
`build-gfx908/install`. It uses the pinned `third_party/triton` compiler in
CMake's isolated build venv, rather than the base image's Triton. Kernel compile
errors fail the build; the GPU compilation timeout is disabled. `MAX_JOBS`
defaults to eight for this build. Set `AOTRITON_TRITON_WHEEL` to an absolute
path to reuse a wheel built from the repository's Triton pin.

Re-run `bash .devcontainer/build.sh` after editing sources; it reconfigures
CMake so generator changes are picked up. When upgrading an existing workspace
from the previous image, start with fresh CMake build directories for the new
Python/ROCm toolchain. The code generator in CMake's venv
is a separate non-editable install refreshed during configuration. Additional
CMake options can be passed to the helper. For example, a tuning build gets
its own directory:

```bash
AOTRITON_BUILD_DIR="$PWD/build-gfx908-tune" \
  bash .devcontainer/build.sh -DAOTRITON_BUILD_FOR_TUNING=ON
```

## Test the local AOTriton build

Expose the installed Python binding, then check that it and the gfx908 kernel
archives exist:

```bash
export PYTHONPATH="$PWD/build-gfx908/install/lib${PYTHONPATH:+:$PYTHONPATH}"
python .devcontainer/doctor.py --aotriton
```

Start with small forward/backward checks across FP16, BF16 and FP32, followed
by a variable-length forward check. These explicitly select the Triton backends
for the gfx908 port:

```bash
FWD_IMPL=triton BWD_IMPL=triton_split python -m pytest -q -x \
  modules/flash/tests/test_backward.py::test_logsumexp_scaling
FWD_IMPL=triton BWD_IMPL=triton_split python -m pytest -q -x \
  modules/flash/tests/test_backward.py::test_matrix_bias_fwd_bwd_symmetry
FWD_IMPL=triton BWD_IMPL=triton_split python -m pytest -q -x \
  modules/flash/tests/test_varlen.py::test_logsumexp_layout
```

Then exercise the broader forward/backward and variable-length suites:

```bash
FWD_IMPL=triton BWD_IMPL=triton_split FOR_RELEASE=0 \
  python -m pytest -q -x modules/flash/tests/test_backward.py modules/flash/tests/test_varlen.py
```

Repeat with `BWD_IMPL=triton_fuse` to investigate the fused backward path.
Leaving `FWD_IMPL` and `BWD_IMPL` unset tests normal dispatcher selection;
forcing a backend suppresses the tuning-database fallback. Run without xdist
initially for the single reserved GPU. CPU code-generator tests are available
with `python -m pytest -q python/test`.

The environment sets `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` for gfx908
experiments. Direct `pyaotriton` tests exercise this checkout. A later PyTorch
SDPA integration still needs a compatible adapter/library pairing; the base
image's built-in SDPA continues to use its packaged AOTriton.

## Image settings

Override `build.args.ROCM_IMAGE` for another ROCm/PyTorch base, and
`build.args.CODEX_VERSION` for another Codex CLI version. The image must provide
ROCm PyTorch and the development SDK for the selected Python/ROCm stack. The
current package commands assume Ubuntu. Dependency installation outside the
base image is not lockfile-pinned.

Coder's editor configuration can provide extensions if Envbuilder does not
apply VS Code customizations. Codex CLI is installed independently in the image,
and its state remains in `/workspaces/.codex` across rebuilds using the same PVC.
