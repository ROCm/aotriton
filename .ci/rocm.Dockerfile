FROM aotriton:base

# The base image defaults to a non-root user; this image's RUNs need root, and the
# default goes back to that user at the end (see base.Dockerfile).
USER root

ARG ROCM_VERSION_IN_URL
COPY dockerscript-setup-repo.sh /root
RUN bash /root/dockerscript-setup-repo.sh ${ROCM_VERSION_IN_URL} && dnf install -y rocm-hip-runtime rocm-hip-runtime-devel hipcc rocm-device-libs hip-devel && dnf clean all

USER builder

# A build environment: nothing listens, so there is nothing to probe.
HEALTHCHECK NONE
