FROM aotriton:base AS base

ARG PYVER
RUN dnf install -y python${PYVER} python${PYVER}-devel && \
    update-alternatives --set python /usr/bin/python${PYVER} && \
    update-alternatives --set python3 /usr/bin/python${PYVER} && \
    python -m ensurepip && \
    python -m pip install cmake ninja wheel
