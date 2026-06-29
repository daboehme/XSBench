cmake_minimum_required(VERSION 3.21)
project(XSBench LANGUAGES C CXX)

# initialize CUDA_ARCHITECTURES from nvcc default
cmake_policy(SET CMP0104 NEW)

find_package(caliper REQUIRED)
find_package(adiak REQUIRED)

option(XSBENCH_CUDA "Build the CUDA version" OFF)
option(XSBENCH_HIP  "Build the HIP version"  OFF)
option(XSBENCH_MPI  "Build with MPI support" OFF)

if (XSBENCH_CUDA)
  if (XSBENCH_MPI)
    message(WARNING "CUDA version does not support MPI")
  endif()
  add_subdirectory(cuda)
elseif(XSBENCH_HIP)
  if (XSBENCH_MPI)
    message(WARNING "HIP version does not support MPI")
  endif()
  add_subdirectory(hip)
else()
  add_subdirectory(openmp-threading)
endif()
