ifeq ($(shell uname -s), Linux)
NVCC = /usr/local/cuda-8.0/bin/nvcc
else ifeq ($(shell uname -s), Darwin)
NVCC = /Developer/NVIDIA/CUDA-8.0/bin/nvcc
else
NVCC = nvcc
endif

FLAGS += -gencode arch=compute_61,code=sm_61

CFLAGS = $(FLAGS)
CFLAGS += -std=c++11
ifdef RELEASE
CFLAGS += -O2
else
CFLAGS += -G -g -O0
endif

CDFLAGS = $(CFLAGS)
CHFLAGS = $(CFLAGS) --compile --relocatable-device-code=false
LDFLAGS = $(FLAGS) --cudart static --relocatable-device-code=false -lcurand -lcublas

rwildcard = $(wildcard $(addsuffix $2, $1)) $(foreach d,$(wildcard $(addsuffix *, $1)),$(call rwildcard,$d/,$2))

BUILDDIR = target/cuda
SOURCES := $(call rwildcard,./,*.cu)
OBJECTS = $(patsubst %.cu,$(BUILDDIR)/%.o,$(SOURCES))
DOBJECTS = $(patsubst %.cu,$(BUILDDIR)/%.d,$(SOURCES))
EXECUTABLE = GPUPolyaUrnLDA
LIBRARY = libGPUPolyaUrnLDA.a

.PHONY: all clean

all: $(BUILDDIR)/$(EXECUTABLE) $(BUILDDIR)/$(LIBRARY)

$(DOBJECTS): $(BUILDDIR)/%.d: %.cu
	mkdir -p $(@D)
	$(NVCC) $(CDFLAGS) -odir "$(@D)" -M -o "$@" "$<"

$(OBJECTS): $(BUILDDIR)/%.o: %.cu $(BUILDDIR)/%.d
	$(NVCC) $(CHFLAGS) -odir "$(@D)" -x cu -o "$@" "$<"

$(BUILDDIR)/$(EXECUTABLE): $(OBJECTS)
	$(NVCC) $(LDFLAGS) -link -o "$@" $(OBJECTS)

$(BUILDDIR)/$(LIBRARY): $(OBJECTS)
	$(NVCC) $(LDFLAGS) -lib -o "$@" $(OBJECTS)

clean:
	rm -rv $(BUILDDIR)/$(EXECUTABLE) $(BUILDDIR)/$(LIBRARY) $(OBJECTS) $(DOBJECTS) $(DEPENDENCIES)
