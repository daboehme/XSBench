set(XSBENCH_SOURCES
  GridInit.c
  io.c
  Main.c
  Materials.c
  Simulation.c
  XSutils.c)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_executable(XSBench ${XSBENCH_SOURCES})

target_include_directories(XSBench PRIVATE ${caliper_INCLUDE_DIR} ${adiak_INCLUDE_DIRS})
target_link_libraries(XSBench PRIVATE caliper ${adiak_LIBRARIES})

find_package(OpenMP)
if (OpenMP_C_Found)
  target_link_libraries(XSBench PRIVATE OpenMP::OpenMP_C)
endif()

if (XSBENCH_MPI)
  find_package(MPI)
  if (MPI_C_FOUND)
    target_link_libraries(XSBench PRIVATE MPI::MPI_C)
  endif()
endif()

install(TARGETS XSBench DESTINATION ${CMAKE_INSTALL_BINDIR})
