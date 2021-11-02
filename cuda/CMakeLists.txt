set(XSBENCH_SOURCES
  GridInit.cu
  io.cu
  Main.cu
  Materials.cu
  Simulation.cu
  XSutils.cu)

enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_executable(XSBench ${XSBENCH_SOURCES})

target_include_directories(XSBench PRIVATE ${caliper_INCLUDE_DIR} ${adiak_INCLUDE_DIRS})
target_link_libraries(XSBench PRIVATE caliper ${adiak_LIBRARIES})

include(CheckLibraryExists)
CHECK_LIBRARY_EXISTS(m ceil "" HAVE_LIBM)

if (HAVE_LIBM)
  list(APPEND EXTRA_LIBS m)
endif()

target_link_libraries(XSBench PRIVATE ${EXTRA_LIBS})

install(TARGETS XSBench DESTINATION ${CMAKE_INSTALL_BINDIR})
