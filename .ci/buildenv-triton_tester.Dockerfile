FROM aotriton:base AS base

ARG PYVER
RUN dnf install -y python${PYVER} python${PYVER}-devel && \
    dnf clean all && \
    update-alternatives --set python /usr/bin/python${PYVER} && \
    update-alternatives --set python3 /usr/bin/python${PYVER} && \
    python -m ensurepip && \
    python -m pip install cmake ninja wheel

# A build environment: nothing listens, so there is nothing to probe.
HEALTHCHECK NONE
