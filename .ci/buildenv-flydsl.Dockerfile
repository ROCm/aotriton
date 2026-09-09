# Build environment for the FlyDSL compiler wheel.
#
# aotriton:base-py<X.Y> is enough to build LLVM, but not FlyDSL: FlyDSL's
# lib/Runtime/ROCm/CMakeLists.txt does an unconditional
#   find_package(hip REQUIRED CONFIG PATHS /opt/rocm*)
# under the only backend it has (FLYDSL_BACKENDS defaults to, and only allows,
# "rocdl"), so a ROCm-less image fails at configure time. This adds the ROCm
# dev packages on top of the base image, exactly as rocm.Dockerfile does for
# the AOTriton build environment.
#
# What is being satisfied is a *link* requirement, not a runtime one:
# FlyJitRuntime is a thin HIP wrapper for launching kernels, which AOTriton
# never does -- it compiles under COMPILE_ONLY=1 and reads the artifact back
# out. So the ROCm version here has no bearing on the kernels the resulting
# wheel produces, and the default is simply the current stable release. No GPU
# is needed to build in this image.

ARG PYVER=3.11
FROM aotriton:base-py${PYVER}

ARG ROCM_VERSION_IN_URL=7.2.4
COPY dockerscript-setup-repo.sh /root
RUN bash /root/dockerscript-setup-repo.sh ${ROCM_VERSION_IN_URL} && \
    dnf install -y rocm-hip-runtime rocm-hip-runtime-devel hipcc rocm-device-libs hip-devel
