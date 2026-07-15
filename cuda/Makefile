#===============================================================================
# User Options
#===============================================================================

COMPILER    = nvidia
OPTIMIZE    = yes
DEBUG       = no
PROFILE     = no
SM_VERSION  = 80
CALI_DIR    = /usr
ADIAK_DIR   = /usr

#===============================================================================
# Program name & source code list
#===============================================================================

program = XSBench

source = \
Main.cu \
io.cu \
Simulation.cu \
GridInit.cu \
XSutils.cu \
Materials.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -I$(CALI_DIR)/include -I$(ADIAK_DIR)/include

# Linker Flags
LDFLAGS = -lm -Wl,-rpath $(CALI_DIR)/lib64 -Wl,-rpath $(ADIAK_DIR)/lib -L$(CALI_DIR)/lib64 -L$(ADIAK_DIR)/lib -lcaliper -ladiak

# NVIDIA Compiler
ifeq ($(COMPILER),nvidia)
  CC = nvcc
  CFLAGS += -Xcompiler -Wall -Xcompiler -O3 -arch=sm_$(SM_VERSION) -std=c++14
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -G
  LDFLAGS  += -g -G
endif

# Profiling Flags
ifeq ($(PROFILE),yes)
  CFLAGS += -pg
  LDFLAGS  += -pg
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) XSbench_header.cuh Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu XSbench_header.cuh Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

edit:
	vim -p $(source) XSbench_header.cuh

run:
	./$(program)
