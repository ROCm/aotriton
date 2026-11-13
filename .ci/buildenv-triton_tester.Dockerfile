FROM aotriton:base AS base

# The base image defaults to a non-root user; this image's RUNs need root, and the
# default goes back to that user at the end (see base.Dockerfile).
USER root

ARG PYVER
RUN dnf install -y python${PYVER} python${PYVER}-devel && \
    dnf clean all && \
    update-alternatives --set python /usr/bin/python${PYVER} && \
    update-alternatives --set python3 /usr/bin/python${PYVER} && \
    python -m ensurepip && \
    python -m pip install cmake ninja wheel

USER builder

# A build environment: nothing listens, so there is nothing to probe.
HEALTHCHECK NONE
