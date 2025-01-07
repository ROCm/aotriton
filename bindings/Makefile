SUFFIX := $(shell python3-config --extension-suffix)
INC := -I../include/ -I../third_party/pybind11/include/ $(shell python3-config --includes)
COPTIONS := -O2 -Wall -std=c++20 -fPIC 

final: hipruntime.o module.o
	hipcc -fPIC -shared hipruntime.o module.o -o pyaotriton$(SUFFIX) libaotriton_v2.so $(shell python3-config --libs) -Wl,-rpath='$$ORIGIN'

hipruntime.o:
	hipcc $(COPTIONS) $(INC) -c ../bindings/hipruntime.cc

# Note this Makefile will be invoked under mbuild directory
module.o:
	hipcc $(COPTIONS) $(INC) -c ../bindings/module.cc
