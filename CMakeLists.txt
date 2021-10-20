cmake_minimum_required(VERSION 3.12)
project(XSBench LANGUAGES C)

find_package(caliper REQUIRED)
find_package(adiak REQUIRED)

option(XSBENCH_CUDA "Build the CUDA version" OFF)
option(XSBENCH_MPI  "Build with MPI support" OFF)

if (XSBENCH_CUDA)
  if (XSBENCH_MPI)
    message(WARNING "CUDA version does not support MPI")
  endif()
  add_subdirectory(cuda)
else()
  add_subdirectory(openmp-threading)
endif()
