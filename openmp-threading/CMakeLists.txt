set(XSBENCH_SOURCES
  GridInit.c
  io.c
  Main.c
  Materials.c
  Simulation.c
  XSutils.c)

set(CMAKE_C_STANDARD 99)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_executable(XSBench ${XSBENCH_SOURCES})

target_include_directories(XSBench PRIVATE ${caliper_INCLUDE_DIR} ${adiak_INCLUDE_DIRS})
target_link_libraries(XSBench PRIVATE caliper ${adiak_LIBRARIES})

include(CheckLibraryExists)
CHECK_LIBRARY_EXISTS(m ceil "" HAVE_LIBM)

if (HAVE_LIBM)
  list(APPEND EXTRA_LIBS m)
endif()

find_package(OpenMP)
if (OpenMP_C_FOUND)
  list(APPEND EXTRA_LIBS OpenMP::OpenMP_C)
endif()

if (XSBENCH_MPI)
  find_package(MPI)
  if (MPI_C_FOUND)
    list(APPEND EXTRA_LIBS MPI::MPI_C)
  endif()
endif()

target_link_libraries(XSBench PRIVATE ${EXTRA_LIBS})

install(TARGETS XSBench DESTINATION ${CMAKE_INSTALL_BINDIR})
