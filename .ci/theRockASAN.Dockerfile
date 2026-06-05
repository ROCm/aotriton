FROM aotriton:base

ARG THEROCK_VERSION=7.14.0.dev0+9dfbd936fa3750b21f4aabea2f861f97a435aec0
ARG THEROCK_PIP_FIND_LINKS=https://therock-ci-artifacts.s3.amazonaws.com/26732801416-linux/python/index.html

# Create the venv at /opt/therock, install rocm wheels into it, then let
# rocm-sdk init copy/link the actual ROCm tree into place. ROCM_PATH is
# derived from `rocm-sdk path --root` after init.
RUN set -ex \
 && python3 -m venv /opt/therock \
 && . /opt/therock/bin/activate \
 && pip install --upgrade pip \
 && pip install \
      "rocm[libraries,devel,device-gfx942,device-gfx950]" --pre \
      --find-links="${THEROCK_PIP_FIND_LINKS}" \
 && rocm-sdk init \
 && rocm_root=$(rocm-sdk path --root) \
 && echo "Resolved ROCM_PATH=${rocm_root}" \
 && printf '%s\n' "${rocm_root}" > /opt/therock/.rocm_root

# Bake ROCM_PATH (and PATH/LD_LIBRARY_PATH) from the recorded root, and
# auto-activate the venv so every `docker run <image> ...` inherits it.
ENV VIRTUAL_ENV=/opt/therock
ENV PATH="/opt/therock/bin:${PATH}"
ENV BASH_ENV=/etc/profile.d/therock.sh

RUN set -ex \
 && rocm_root=$(cat /opt/therock/.rocm_root) \
 && mkdir -p /etc/profile.d \
 && cat > /etc/profile.d/therock.sh <<EOF
# Auto-activate the theRock venv and export ROCM_PATH for every shell.
. /opt/therock/bin/activate
export ROCM_PATH="${rocm_root}"
export PATH="\${ROCM_PATH}/bin:\${ROCM_PATH}/llvm/bin:\${PATH}"
export LD_LIBRARY_PATH="\${ROCM_PATH}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
EOF
