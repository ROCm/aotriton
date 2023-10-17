## Usage

```
mkdir build
python python/generate.py
(cd build; make -j `nproc` -f Makefile.compile)
python python/generate_shim.py
(cd build; make -j `nproc` -f Makefile.shim)
```

Then the `attn_fwd.so` and `attn_fwd.h` can be found under `build/`

### Prerequisites

* `hipcc`
* `triton`
