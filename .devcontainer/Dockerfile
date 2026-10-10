# Build context is only .devcontainer: repo contents and credentials are not
# copied into the image. Envbuilder clones the repo onto the workspace PVC.
ARG ROCM_IMAGE=rocm/pytorch:rocm10.1.0_ubuntu26.04_py3.14_pytorch_release_2.14.0@sha256:68295d40dcdcb53d9b50828f30017faad8cf5ad9a8c7923c5a815b414b45e3e6
FROM ${ROCM_IMAGE}

USER root
ARG CODEX_VERSION=0.154.0
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential ca-certificates ccache cmake curl gdb git git-lfs jq \
        less libedit-dev libffi-dev liblzma-dev libxml2-dev libzstd-dev mold \
        ninja-build nodejs npm openssh-client openssl patchelf pkg-config \
        python3-dev python3-venv ripgrep rsync shellcheck sudo tmux unzip vim \
        wget xz-utils zip zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g "@openai/codex@${CODEX_VERSION}" \
    && npm cache clean --force

# Development sessions run as root. Codex state lives on the workspace PVC.
RUN mkdir -p /workspaces

# Keep the ROCm image's PyTorch for tensor allocation and test references.
# AOTriton's CMake build uses a separate venv and its own pinned Triton compiler.
# Development packages go into a writable venv rebuilt with the image.
RUN python3 -c 'import site; print("\n".join(site.getsitepackages()))' > /tmp/rocm-python-sites \
    && python3 -m venv --system-site-packages /opt/aotriton-venv \
    && cp /tmp/rocm-python-sites "$(/opt/aotriton-venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/rocm-base.pth" \
    && rm /tmp/rocm-python-sites \
    && /opt/aotriton-venv/bin/python -m pip install --no-cache-dir \
        'cmake>=3.30,<4' 'setuptools>=77,<81' packaging wheel pybind11

ENV PATH=/opt/aotriton-venv/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:${PATH} \
    VIRTUAL_ENV=/opt/aotriton-venv \
    CODEX_HOME=/workspaces/.codex \
    ROCM_PATH=/opt/rocm \
    LLVM_TARBALL_DISTRO=ubuntu26.04 \
    AOTRITON_TARGET_ARCH=gfx908 \
    PYTORCH_ROCM_ARCH=gfx908 \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    MAX_JOBS=8

# These checks need no GPU and fail image creation if essential tools are absent.
RUN command -v hipcc && command -v llvm-objdump && command -v llvm-mc \
    && command -v ld.lld && command -v ninja && command -v pkg-config \
    && pkg-config --exists liblzma && codex --version \
    && python -c 'import torch; assert torch.version.hip, "ROCm PyTorch required"'

WORKDIR /workspaces
USER root
CMD ["bash"]
