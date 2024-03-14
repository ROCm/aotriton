NPROC=$(shell nproc)

v2:
	mkdir -p mbuild # Manual build
	python -m v2python.generate_compile --build_dir mbuild --target_gpus MI200
	(. mbuild/venv/bin/activate; cd mbuild; LD_PRELOAD=/opt/rocm/lib/libamdocl64.so make -j $(NPROC) -f Makefile.compile)
	python -m v2python.generate_shim --build_dir mbuild --target_gpus MI200 --build_dir mbuild
	# (. mbuild/venv/bin/activate; cd mbuild/flash/autotune.attn_fwd; hipcc -std=c++20 -c -I../../../include -I../../../third_party/incbin/ 'FONLY__^bf16@16,1,128,False,True___MI200.cc')
	(. mbuild/venv/bin/activate; cd mbuild; make -j $(NPROC) -f Makefile.shim EXTRA_COMPILER_OPTIONS="-O2 -DNDEBUG")

v2binding:
	(. mbuild/venv/bin/activate; cd mbuild; make final -j2 -f ../bindings/Makefile)

check:
	nm -DC build/libaotriton_v2.so |grep aotriton|grep 'U '

test:
	PYTHONPATH=39build/bindings/ pytest -s test/test_forward.py

all:
	mkdir -p build
	python python/generate.py --target MI200
	(. build/venv/bin/activate; cd build; LD_PRELOAD=/opt/rocm/lib/libamdocl64.so make -j $(NPROC) -f Makefile.compile)
	python python/generate_shim.py
	(. build/venv/bin/activate; cd build; make -j $(NPROC) -f Makefile.shim)

format:
	find bindings/ include/ v2src/ \( -name '*.h' -or -name '*.cc' \) -not -path '*template/*' -exec clang-format -i {} \;

test_compile:
	hipcc -o build/test_compile test/test_compile.cc -L build -laotriton -Wl,-rpath=. -I/opt/rocm/include -Ibuild/

clean:
	(cd mbuild/; rm -f *.h *.so *.cc *.o *.json *.hsaco)

create_venv:
	python -m venv mbuild/venv

triton_install_develop:
	(. mbuild/venv/bin/activate; pip install -r requirements.txt; cd third_party/triton/python/; pip install -e .)

triton_install:
	(. mbuild/venv/bin/activate; pip install -r requirements.txt; cd third_party/triton/python/; pip install .)

.PHONY: all clean test_compile create_venv triton_install triton_install_develop check test
