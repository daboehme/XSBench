#===============================================================================
# User Options
#===============================================================================

COMPILER    ?= gnu
OPTIMIZE    ?= yes
DEBUG       ?= no
PROFILE     ?= no
USE_MPI     ?= no

#===============================================================================
# Program name & source code list
#===============================================================================

program = XSBench

source = \
Main.c \
io.c \
Simulation.c \
GridInit.c \
XSutils.c \
CLutils.c \
Materials.c

obj = $(source:.c=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -std=gnu99 -Wall

# Preserve compatibility with external tooling that still passes MPI=yes/no.
ifeq ($(MPI),yes)
USE_MPI = yes
endif
ifeq ($(MPI),no)
USE_MPI = no
endif

# Linker Flags
LDFLAGS = -lm -lOpenCL

# Regular gcc Compiler
ifeq ($(COMPILER),gnu)
  CC = gcc
endif

# clang
ifeq ($(COMPILER),clang)
  CC = clang
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
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

# MPI
ifeq ($(USE_MPI),yes)
  CC = mpicc
  CFLAGS += -DUSE_MPI
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) XSbench_header.h CLutils.h Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.c XSbench_header.h CLutils.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

edit:
	vim -p $(source) kernel.cl XSbench_header.h CLutils.h

run:
	./$(program)
