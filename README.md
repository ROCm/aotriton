## Build Instructions

```
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=./install_dir
# Use ccmake to tweak options
make install
```

The library and the header file can be found under `build/install_dir` afterwards.

Note: do not run `make` separately, due to the limit of the current build
system, `make install` will run the whole build process unconditionally.

### Prerequisites

* `hipcc` in `/opt/rocm/bin`, as a part of [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
