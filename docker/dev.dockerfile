ARG baseimage=rocm/dev-manylinux2014_x86_64:6.0.2
ARG commit=main

FROM $baseimage
RUN yum install -y ninja-build
WORKDIR /usr/src
RUN git clone --recurse-submodules --shallow-submodules https://github.com/ROCm/aotriton.git
RUN mkdir aotriton/build
WORKDIR /usr/src/aotriton/build
RUN cmake .. -DCMAKE_INSTALL_PREFIX=/opt/aotriton -DCMAKE_BUILD_TYPE=Release -DAOTRITON_COMPRESS_KERNEL=OFF -DAOTRITON_NO_PYTHON=ON
RUN make install
RUN cp /usr/src/aotriton/LICENSE /opt/aotriton/
