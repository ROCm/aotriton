NPROC=$(shell nproc)

v2all:
	mkdir -p build
	python -m v2python.generate_compile --target_gpus MI200
	(. build/venv/bin/activate; cd build; LD_PRELOAD=/opt/rocm/lib/libamdocl64.so make -j $(NPROC) -f Makefile.compile)
	python -m v2python.generate_shim --target_gpus MI200 --archive --build_dir build
	# (. build/venv/bin/activate; cd build/flash/; hipcc -std=c++20 -c -I../../include -I../../third_party/incbin/ attn_fwd.cc)
	# (. build/venv/bin/activate; cd build/flash/autotune.attn_fwd; hipcc -std=c++20 -c -I../../../include -I../../../third_party/incbin/ 'FONLY__^bf16@16,1,128,False,True___MI200.cc')
	(. build/venv/bin/activate; cd build; make -j $(NPROC) -f Makefile.shim)

all:
	mkdir -p build
	python python/generate.py --target MI200
	(. build/venv/bin/activate; cd build; LD_PRELOAD=/opt/rocm/lib/libamdocl64.so make -j $(NPROC) -f Makefile.compile)
	python python/generate_shim.py
	(. build/venv/bin/activate; cd build; make -j $(NPROC) -f Makefile.shim)

test_compile:
	hipcc -o build/test_compile test/test_compile.cc -L build -laotriton -Wl,-rpath=. -I/opt/rocm/include -Ibuild/

clean:
	(cd build/; rm -f *.h *.so *.cc *.o *.json *.hsaco)

create_venv:
	python -m venv build/venv

triton_install_develop:
	(. build/venv/bin/activate; pip install -r requirements.txt; cd third_party/triton/python/; pip install -e .)

triton_install:
	(. build/venv/bin/activate; pip install -r requirements.txt; cd third_party/triton/python/; pip install .)

.PHONY: all clean test_compile create_venv triton_install triton_install_develop
