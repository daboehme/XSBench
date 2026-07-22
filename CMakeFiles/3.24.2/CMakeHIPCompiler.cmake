set(CMAKE_HIP_COMPILER "/usr/lib64/ccache/clang++")
set(CMAKE_HIP_COMPILER_ID "Clang")
set(CMAKE_HIP_COMPILER_VERSION "21.1.8")
set(CMAKE_HIP_STANDARD_COMPUTED_DEFAULT "17")
set(CMAKE_HIP_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_HIP_COMPILE_FEATURES "hip_std_98;hip_std_11;hip_std_14;hip_std_17;hip_std_20;hip_std_23")
set(CMAKE_HIP98_COMPILE_FEATURES "")
set(CMAKE_HIP11_COMPILE_FEATURES "hip_std_11")
set(CMAKE_HIP14_COMPILE_FEATURES "hip_std_14")
set(CMAKE_HIP17_COMPILE_FEATURES "hip_std_17")
set(CMAKE_HIP20_COMPILE_FEATURES "hip_std_20")
set(CMAKE_HIP23_COMPILE_FEATURES "hip_std_23")

set(CMAKE_HIP_PLATFORM_ID "Linux")
set(CMAKE_HIP_SIMULATE_ID "")
set(CMAKE_HIP_COMPILER_FRONTEND_VARIANT "GNU")
set(CMAKE_HIP_SIMULATE_VERSION "")


set(CMAKE_HIP_COMPILER_ROCM_ROOT "/opt/rocm-7.2.1")

set(CMAKE_HIP_COMPILER_ENV_VAR "HIPCXX")

set(CMAKE_HIP_COMPILER_LOADED 1)
set(CMAKE_HIP_COMPILER_ID_RUN 1)
set(CMAKE_HIP_SOURCE_FILE_EXTENSIONS hip)
set(CMAKE_HIP_LINKER_PREFERENCE 90)
set(CMAKE_HIP_LINKER_PREFERENCE_PROPAGATES 1)

set(CMAKE_HIP_SIZEOF_DATA_PTR "8")
set(CMAKE_HIP_COMPILER_ABI "ELF")
set(CMAKE_HIP_LIBRARY_ARCHITECTURE "x86_64-redhat-linux-gnu")

if(CMAKE_HIP_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_HIP_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_HIP_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_HIP_COMPILER_ABI}")
endif()

if(CMAKE_HIP_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "x86_64-redhat-linux-gnu")
endif()

set(CMAKE_HIP_TOOLKIT_INCLUDE_DIRECTORIES "")

set(CMAKE_HIP_IMPLICIT_INCLUDE_DIRECTORIES "/opt/rocm-7.2.1/include;/usr/lib/clang/21/include/cuda_wrappers;/opt/rh/gcc-toolset-15/root/usr/include/c++/15;/opt/rh/gcc-toolset-15/root/usr/include/c++/15/x86_64-redhat-linux;/opt/rh/gcc-toolset-15/root/usr/include/c++/15/backward;/usr/lib/clang/21/include;/usr/local/include;/usr/include")
set(CMAKE_HIP_IMPLICIT_LINK_LIBRARIES "amdhip64;stdc++;m;gcc_s;c;gcc_s")
set(CMAKE_HIP_IMPLICIT_LINK_DIRECTORIES "/usr/lib/clang/21/lib/x86_64-redhat-linux-gnu;/opt/rh/gcc-toolset-15/root/usr/lib/gcc/x86_64-redhat-linux/15;/opt/rh/gcc-toolset-15/root/usr/lib64;/lib64;/usr/lib64;/lib;/usr/lib;/opt/rocm-7.2.1/lib")
set(CMAKE_HIP_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_HIP_RUNTIME_LIBRARY_DEFAULT "SHARED")

set(CMAKE_AR "/usr/bin/llvm-ar")
set(CMAKE_HIP_COMPILER_AR "/usr/bin/llvm-ar-21")
set(CMAKE_RANLIB "/usr/bin/llvm-ranlib")
set(CMAKE_HIP_COMPILER_RANLIB "/usr/bin/llvm-ranlib-21")
set(CMAKE_LINKER "/usr/bin/ld.lld")
set(CMAKE_MT "")
