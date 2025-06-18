# TL;DR

``` bash
mkdir -p build-test
cd build-test
# AOTRITON_NAME_SUFFIX is essential to avoid symbol conflicts with AOTriton bundled in PyTorch
cmake .. -DCMAKE_INSTALL_PREFIX=./install_dir -DCMAKE_BUILD_TYPE=Release -DAOTRITON_GPU_BUILD_TIMEOUT=0 -G Ninja -DAOTRITON_NAME_SUFFIX=123
# Optionally only build for one arch
# cmake .. -DCMAKE_INSTALL_PREFIX=./install_dir -DCMAKE_BUILD_TYPE=Release -DAOTRITON_GPU_BUILD_TIMEOUT=0 -G Ninja -DAOTRITON_NAME_SUFFIX=123 -DAOTRITON_TARGET_ARCH=gfx942
ninja install
FOR_RELEASE=1 PYTHONPATH=install_dir/lib/ pytest ../test/test_backward.py -v
# Optionally starting from AOTriton >= 0.10.0
# it is possible to run tests in parallel on multi-GPUs
# NGPUS=$(amd-smi list --json|jq length)
# FOR_RELEASE=1 PYTHONPATH=install_dir/lib/ pytest -n $NGPUS ../test/test_backward.py -v
```

# Pre-requisites

* `pip install -r requirements-dev.txt`

# Pre-requisites for parallel testing on multi-GPUs

* AOTriton >= 0.10.0
* `jq` (usually available as `apt install jq` or `dnf install jq`)
* Python CLI `amd-smi`. Install with https://rocm.docs.amd.com/projects/amdsmi/en/latest/install/install.html#manually-install-the-python-library if not available in current python venv.
