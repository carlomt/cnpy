cmake_minimum_required(VERSION 3.12...3.27)

# Modern CMake, no need for legacy policies
project(CNPY VERSION 0.2.0 LANGUAGES CXX)

include(GNUInstallDirs)

option(CNPY_BUILD_SHARED "Build shared library" ON)
option(CNPY_BUILD_STATIC "Build static library" ON)
option(CNPY_BUILD_EXAMPLES "Build example programs" OFF)

# Require C++11 features instead of forcing global compiler flags
add_library(cnpy_compile_features INTERFACE)
target_compile_features(cnpy_compile_features INTERFACE cxx_std_11)

find_package(ZLIB REQUIRED)

# Common sources
set(CNPY_SOURCES cnpy.cpp)
set(CNPY_PUBLIC_HEADERS cnpy.h numpy_defs.h)

# Helper function for shared/static libraries
function(cnpy_common_setup tgt)
  target_link_libraries(${tgt}
    PRIVATE
      ZLIB::ZLIB
  )
  target_include_directories(${tgt}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
  # Ensure PIC (Position Independent Code) even for static builds
  # This avoids linker errors when linking into shared libraries (e.g. Geant4)
  set_target_properties(${tgt} PROPERTIES POSITION_INDEPENDENT_CODE ON)

  # Propagate C++ standard requirements
  target_link_libraries(${tgt} PUBLIC cnpy_compile_features)
endfunction()

if(CNPY_BUILD_SHARED)
  add_library(cnpy SHARED ${CNPY_SOURCES})
  cnpy_common_setup(cnpy)

  # Default visibility is fine; no export header needed here
  install(TARGETS cnpy
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()

if(CNPY_BUILD_STATIC)
  add_library(cnpy-static STATIC ${CNPY_SOURCES})
  # Use a distinct name for the static library to avoid overwriting
  set_target_properties(cnpy-static PROPERTIES OUTPUT_NAME "cnpy_static")
  cnpy_common_setup(cnpy-static)

  install(TARGETS cnpy-static
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()

# Public headers
install(FILES ${CNPY_PUBLIC_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Optional examples
if(CNPY_BUILD_EXAMPLES)
  add_executable(example1 example1.cpp)
  # Link against the shared library if available, otherwise static
  if(TARGET cnpy)
    target_link_libraries(example1 PRIVATE cnpy)
  else()
    target_link_libraries(example1 PRIVATE cnpy-static)
  endif()
  install(TARGETS example1 RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# NOTE:
# - Removed the following install command:
#   install(FILES "mat2npz" "npy2mat" "npz2mat" DESTINATION bin)
#   because these are not compiled executables. 
#   If they are meant to be scripts, install them explicitly later.
