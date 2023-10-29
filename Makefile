all:
	mkdir -p build
	python python/generate.py
	(cd build; make -j $(shell nproc) -f Makefile.compile)
	python python/generate_shim.py
	(cd build; make -j $(shell nproc) -f Makefile.shim)

test_compile:
	hipcc -o build/test_compile test/test_compile.cc -L build -loort -Wl,-rpath=. -I/opt/rocm/include -Ibuild/

clean:
	(cd build/; rm -f *.h *.so *.cc *.o *.json *.hsaco)

.PHONY: all clean test_compile
