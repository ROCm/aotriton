FROM almalinux:8 AS base

ARG PYVER=3.11
RUN dnf install -y gcc-toolset-13 python${PYVER} python${PYVER}-devel libatomic \
    zstd libzstd-devel xz-devel zlib-devel git which vim wget rsync && \
    update-alternatives --set python /usr/bin/python${PYVER} && \
    update-alternatives --set python3 /usr/bin/python${PYVER} && \
    python -m ensurepip && \
    python -m pip install cmake ninja wheel
