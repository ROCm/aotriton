FROM aotriton:base

# aotriton:base ends as the non-root nobody (65534); installing needs root.
USER root

ARG ROCM_VERSION_IN_URL
COPY dockerscript-setup-repo.sh /root
RUN bash /root/dockerscript-setup-repo.sh ${ROCM_VERSION_IN_URL} && dnf install -y rocm-hip-runtime rocm-hip-runtime-devel hipcc rocm-device-libs hip-devel && dnf clean all

# A build environment: nothing listens, so there is nothing to probe.
HEALTHCHECK NONE

USER 65534:65534
