FROM almalinux:8 AS base

RUN dnf install -y gcc-toolset-13 python3.11 python3.11-devel \
    zstd libzstd-devel xz-devel zlib-devel git which vim wget rsync && \
    update-alternatives --set python /usr/bin/python3.11 && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    python -m ensurepip && \
    python -m pip install cmake ninja wheel
