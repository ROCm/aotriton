NPROC=$(shell nproc)

all:
	mkdir -p build
	python python/generate.py --target MI200
	(cd build; LD_PRELOAD=/opt/rocm/lib/libamdocl64.so make -j $(NPROC) -f Makefile.compile)
	python python/generate_shim.py
	(cd build; make -j $(NPROC) -f Makefile.shim)

test_compile:
	hipcc -o build/test_compile test/test_compile.cc -L build -loort -Wl,-rpath=. -I/opt/rocm/include -Ibuild/

clean:
	(cd build/; rm -f *.h *.so *.cc *.o *.json *.hsaco)

create_venv:
	python -m venv build/venv

triton_install:
	(. build/venv/bin/activate; cd third_party/triton/python/; pip install -e .)

.PHONY: all clean test_compile create_venv triton_install
