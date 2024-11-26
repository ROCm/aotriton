#!/bin/bash

AOTRITON_GIT_NAME="$1"
AOTRITON_TARGET_GPUS="$2"
NOIMAGE_MODE="$3"

cd /root/build
(
  git clone --recursive https://github.com/ROCm/aotriton.git
  cd aotriton
  git checkout ${AOTRITON_GIT_NAME}
  git submodule sync
  git submodule update --init --recursive --force 
)
cd /root/build/aotriton/
for f in /input/patch-${AOTRITON_GIT_NAME}/*.patch
do
  if [ -f "$f" ]; then
    echo "apply patch $f"
    git apply "$f"
  fi
done

cd /root/build
if [ "$NOIMAGE_MODE" = "OFF" ]; then
  scl enable gcc-toolset-13 "cd aotriton; python -m pip install -r requirements.txt; cd aotriton/third_party/triton/python; python setup.py develop --user"
else
  scl enable gcc-toolset-13 "cd aotriton; python -m pip install -r requirements.txt"
fi
scl enable gcc-toolset-13 "mkdir -p aotriton/build; cd aotriton/build; cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm -DPYTHON_EXECUTABLE=/usr/bin/python3.11 -DCMAKE_INSTALL_PREFIX=installed_dir/aotriton -DCMAKE_BUILD_TYPE=Release -DAOTRITON_GPU_BUILD_TIMEOUT=0 \"-DTARGET_GPUS=${AOTRITON_TARGET_GPUS}\" -DAOTRITON_NO_PYTHON=ON -DAOTRITON_NOIMAGE_MODE=${NOIMAGE_MODE} -G Ninja && ninja install"
rocmver=$(scl enable gcc-toolset-13 "cpp -I/opt/rocm/include /input/print_rocm_version.h"|tail -n 1|sed 's/ //g')
cd /root/build/aotriton/build/installed_dir && tar cz aotriton > /output/aotriton-${AOTRITON_GIT_NAME}-manylinux_2_28_x86_64-rocm${rocmver}-shared.tar.gz
