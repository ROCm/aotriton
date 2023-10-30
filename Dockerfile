FROM rocm/pytorch

ENV WORKSPACE_DIR=/workspace
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR

RUN git clone https://github.com/ROCmSoftwarePlatform/triton.git \
    && cd triton \
    && git checkout 821e75a2b025c62e4fab0578e32a12b5ca5fc9e9 \
    && echo $PWD \
    && git submodule update --init --recursive \
    && cd python \
    && pip install -e .

RUN git clone https://github.com/ROCmSoftwarePlatform/oort.git \
    && cd oort \
    && git submodule update --init --recursive \
    && mkdir build \
    && python python/generate.py \
    && cd build \
    && make -j `nproc` -f Makefile.compile \
    && cd .. \
    && python python/generate_shim.py \
    && cd build \
    && make -j `nproc` -f Makefile.shim