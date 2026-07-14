#===============================================================================
# User Options
#===============================================================================

COMPILER    = amd
OPTIMIZE    = yes
DEBUG       = no
PROFILE     = no
CALI_DIR    = /usr
ADIAK_DIR   = /usr

#===============================================================================
# Program name & source code list
#===============================================================================

program = XSBench

source = \
Main.cpp \
io.cpp \
Simulation.cpp \
GridInit.cpp \
XSutils.cpp \
Materials.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS += -std=c++14 -I$(CALI_DIR)/include -I$(ADIAK_DIR)/include

# Linker Flags
LDFLAGS = -lm -L$(CALI_DIR)/lib64 -L$(ADIAK_DIR)/lib -lcaliper -ladiak

# AMD
ifeq ($(COMPILER),amd)
  CC = hipcc
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

$(program): $(obj) XSbench_header.h Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp XSbench_header.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

edit:
	vim -p $(source) XSbench_header.h

run:
	./$(program)
