FROM almalinux:8 AS buildenv

RUN dnf install -y gcc-toolset-13 python3.11 python3.11-devel \
    zstd libzstd-devel xz-devel zlib-devel git which vim wget && \
    update-alternatives --set python /usr/bin/python3.11 && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    python -m ensurepip && \
    python -m pip install cmake ninja wheel
RUN mkdir /root/build

ARG amdgpu_installer="https://repo.radeon.com/amdgpu-install/6.2.2/el/8.10/amdgpu-install-6.2.60202-1.el8.noarch.rpm"
ARG amdgpu_installer_select_version="true"
RUN yum install -y "${amdgpu_installer}" && bash -c "${amdgpu_installer_select_version}" && amdgpu-install --usecase=hip --no-dkms --no-32 -y && dnf install -y hipcc rocm-device-libs hip-devel
