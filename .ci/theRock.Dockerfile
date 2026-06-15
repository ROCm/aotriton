FROM aotriton:base

ARG THEROCK_VERSION=7.13.0rc2
ARG THEROCK_BASE_URL=https://rocm.prereleases.amd.com/tarball-multi-arch/

# File name pattern (kept separate from base so the URL is composed here):
#   therock-dist-linux-multiarch-${THEROCK_VERSION}.tar.gz
RUN set -ex \
 && fname="therock-dist-linux-multiarch-${THEROCK_VERSION}.tar.gz" \
 && url="${THEROCK_BASE_URL}${fname}" \
 && mkdir -p /opt/therock \
 && curl -fL "${url}" -o "/tmp/${fname}" \
 && tar -xzf "/tmp/${fname}" -C /opt/therock \
 && rm -f "/tmp/${fname}"

ENV ROCM_PATH=/opt/therock
ENV PATH="${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${PATH}"
ENV LD_LIBRARY_PATH="${ROCM_PATH}/lib:${LD_LIBRARY_PATH}"
