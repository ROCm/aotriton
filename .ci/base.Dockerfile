FROM almalinux:8 AS base

ARG PYVER=3.11

# Python >= 3.13 isn't packaged in AlmaLinux 8's AppStream repos (verified via
# `dnf list --all python3.13`) -- build it from source instead, using
# `make altinstall` so it never overwrites the system python. Every CPython
# build dependency below is already in AlmaLinux 8's baseos repo, so this
# needs no extra repo enablement. PYVER values dnf already packages (e.g.
# 3.11, 3.12) keep using the original dnf install path.
#
# The exact PYVER.z patch is resolved dynamically (latest listed under
# python.org/ftp/python/) rather than pinned, so this Dockerfile never goes
# stale -- the image tag is "-py${PYVER}" (no patch suffix), so once built,
# the existing "build if missing" caching in build_triton_wheels.sh reuses
# whatever patch was resolved at first build, same as any other ARG here.
RUN set -ex; \
    if printf '%s\n%s\n' "3.13" "${PYVER}" | sort -V -C; then \
      dnf install -y gcc-toolset-13 libatomic \
        zstd libzstd-devel xz-devel zlib-devel git which vim wget rsync tar \
        make gcc gcc-c++ \
        openssl-devel bzip2-devel libffi-devel sqlite-devel \
        ncurses-devel readline-devel gdbm-devel libuuid-devel && \
      pyver_full=$(wget -qO- https://www.python.org/ftp/python/ \
        | grep -oE "${PYVER}\.[0-9]+/" | tr -d '/' | sort -V -u | tail -1) && \
      if [ -z "${pyver_full}" ]; then \
        echo "Could not resolve latest ${PYVER}.x release from python.org" >&2; exit 1; \
      fi && \
      cd /usr/src && \
      wget -q "https://www.python.org/ftp/python/${pyver_full}/Python-${pyver_full}.tgz" && \
      tar xzf "Python-${pyver_full}.tgz" && \
      cd "Python-${pyver_full}" && \
      ./configure --with-ensurepip=install && \
      make -j"$(nproc)" && \
      make altinstall && \
      cd / && rm -rf "/usr/src/Python-${pyver_full}"* && \
      update-alternatives --install /usr/bin/python python "/usr/local/bin/python${PYVER}" 1 && \
      update-alternatives --install /usr/bin/python3 python3 "/usr/local/bin/python${PYVER}" 1 && \
      update-alternatives --set python "/usr/local/bin/python${PYVER}" && \
      update-alternatives --set python3 "/usr/local/bin/python${PYVER}"; \

    else \
      dnf install -y gcc-toolset-13 "python${PYVER}" "python${PYVER}-devel" libatomic \
        zstd libzstd-devel xz-devel zlib-devel git which vim wget rsync && \
      update-alternatives --set python "/usr/bin/python${PYVER}" && \
      update-alternatives --set python3 "/usr/bin/python${PYVER}"; \
    fi && \
    python -m ensurepip && \
    python -m pip install cmake ninja wheel
