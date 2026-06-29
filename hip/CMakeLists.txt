set(XSBENCH_SOURCES
  GridInit.cpp
  io.cpp
  Main.cpp
  Materials.cpp
  Simulation.cpp
  XSutils.cpp)

enable_language(HIP)
find_package(hip REQUIRED)

set(CMAKE_C_STANDARD    99)
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set_source_files_properties(Simulation.cpp PROPERTIES LANGUAGE HIP)

add_executable(XSBench ${XSBENCH_SOURCES})
target_link_libraries(XSBench PRIVATE hip::host caliper ${adiak_LIBRARIES})

include(CheckLibraryExists)
CHECK_LIBRARY_EXISTS(m ceil "" HAVE_LIBM)

if (HAVE_LIBM)
  list(APPEND EXTRA_LIBS m)
endif()

target_link_libraries(XSBench PRIVATE ${EXTRA_LIBS})

install(TARGETS XSBench DESTINATION ${CMAKE_INSTALL_BINDIR})
