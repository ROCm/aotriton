all:
	mkdir -p build
	python python/generate.py
	(cd build; make -j $(shell nproc) -f Makefile.compile)
	python python/generate_shim.py
	(cd build; make -j $(shell nproc) -f Makefile.shim)

clean:
	(cd build/; rm -f *.h *.so *.cc *.o *.json *.hsaco)

.PHONY: all clean
