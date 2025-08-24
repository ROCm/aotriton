FROM aotriton:base

ARG ROCM_VERSION_IN_URL
COPY dockerscript-setup-repo.sh /root
RUN bash /root/dockerscript-setup-repo.sh ${ROCM_VERSION_IN_URL} && dnf install -y rocm-hip-runtime hipcc rocm-device-libs hip-devel
