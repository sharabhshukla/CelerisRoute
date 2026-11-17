cmake_minimum_required(VERSION 3.21...4.0)

project(celerisroute LANGUAGES CXX)

option(CELERISROUTE_BUILD_PYTHON "Build the Python extension module" ON)
option(CELERISROUTE_BUILD_BENCHMARKS "Build the native C++ benchmark harness" OFF)
set(CELERISROUTE_WITH_TBB "AUTO" CACHE STRING
  "oneTBB work-stealing scheduler for PricingFleet: AUTO (use if found), ON (require), OFF")
set_property(CACHE CELERISROUTE_WITH_TBB PROPERTY STRINGS AUTO ON OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Boost. Only header-only parts are used (Boost.Graph's dag_shortest_paths and
# r_c_shortest_paths plus their dependencies), so no Boost binary is linked.
# Point CMake at a non-standard install with -DBOOST_ROOT=... or
# -DCMAKE_PREFIX_PATH=...; nothing here hardcodes a path.
# ---------------------------------------------------------------------------
add_library(celerisroute_boost INTERFACE)
find_package(Boost 1.74 CONFIG QUIET)
if(Boost_FOUND AND TARGET Boost::headers)
  target_link_libraries(celerisroute_boost INTERFACE Boost::headers)
  message(STATUS "celerisroute: using Boost ${Boost_VERSION} (CMake package)")
else()
  # Hints may come from the environment with native (backslash) separators.
  set(CELERISROUTE_BOOST_HINTS)
  foreach(hint IN ITEMS "${BOOST_ROOT}" "$ENV{BOOST_ROOT}" "${Boost_INCLUDE_DIR}" "$ENV{BOOST_INCLUDEDIR}")
    if(NOT hint STREQUAL "")
      file(TO_CMAKE_PATH "${hint}" hint)
      list(APPEND CELERISROUTE_BOOST_HINTS "${hint}")
    endif()
  endforeach()
  find_path(CELERISROUTE_BOOST_INCLUDE_DIR
    NAMES boost/graph/r_c_shortest_paths.hpp
    HINTS ${CELERISROUTE_BOOST_HINTS}
    PATH_SUFFIXES include)
  if(NOT CELERISROUTE_BOOST_INCLUDE_DIR)
    message(FATAL_ERROR
      "celerisroute needs the Boost headers (Boost.Graph), which were not found.\n"
      "Install Boost (e.g. 'apt install libboost-dev', 'brew install boost', "
      "'conda install libboost-headers' or 'vcpkg install boost-graph') or pass "
      "-DBOOST_ROOT=<prefix containing include/boost>. With pip:\n"
      "  pip install . --config-settings=cmake.define.BOOST_ROOT=<prefix>")
  endif()
  target_include_directories(celerisroute_boost SYSTEM INTERFACE ${CELERISROUTE_BOOST_INCLUDE_DIR})
  message(STATUS "celerisroute: using Boost headers in ${CELERISROUTE_BOOST_INCLUDE_DIR}")
endif()

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# oneTBB (optional). PricingFleet schedules vessel pricing tasks on a
# tbb::task_arena when it is available and falls back to a plain std::thread
# work queue otherwise, so the library builds and passes its tests without it.
# Install with 'brew install tbb', 'apt install libtbb-dev',
# 'conda install tbb-devel' or 'vcpkg install tbb'; point CMake at a
# non-standard prefix with -DTBB_DIR=<prefix>/lib/cmake/TBB.
# ---------------------------------------------------------------------------
string(TOUPPER "${CELERISROUTE_WITH_TBB}" CELERISROUTE_WITH_TBB_MODE)
set(CELERISROUTE_HAS_TBB OFF)
if(NOT CELERISROUTE_WITH_TBB_MODE MATCHES "^(OFF|0|NO|FALSE)$")
  find_package(TBB CONFIG QUIET COMPONENTS tbb)
  if(TARGET TBB::tbb)
    set(CELERISROUTE_HAS_TBB ON)
    message(STATUS "celerisroute: using oneTBB ${TBB_VERSION} for fleet scheduling")
  elseif(CELERISROUTE_WITH_TBB_MODE MATCHES "^(ON|1|YES|TRUE)$")
    message(FATAL_ERROR
      "CELERISROUTE_WITH_TBB=ON, but oneTBB was not found. Install it (e.g. 'brew install tbb', "
      "'apt install libtbb-dev', 'conda install tbb-devel') or pass "
      "-DTBB_DIR=<prefix>/lib/cmake/TBB. With pip:\n"
      "  pip install . --config-settings=cmake.define.TBB_DIR=<prefix>/lib/cmake/TBB")
  else()
    message(STATUS "celerisroute: oneTBB not found; PricingFleet uses the std::thread scheduler")
  endif()
endif()

# ---------------------------------------------------------------------------
# Solver core, independent of Python.
# ---------------------------------------------------------------------------
add_library(celerisroute_core STATIC
  cpp/src/graph.cpp
  cpp/src/dag_solver.cpp
  cpp/src/rcsp_solver.cpp
  cpp/src/pricing_engine.cpp
  cpp/src/pricing_fleet.cpp)
target_include_directories(celerisroute_core PUBLIC cpp/include)
target_link_libraries(celerisroute_core PUBLIC celerisroute_boost Threads::Threads)
if(CELERISROUTE_HAS_TBB)
  target_link_libraries(celerisroute_core PUBLIC TBB::tbb)
  target_compile_definitions(celerisroute_core PUBLIC CELERISROUTE_HAS_TBB=1)
endif()
target_compile_definitions(celerisroute_core PUBLIC BOOST_ALL_NO_LIB)
if(MSVC)
  target_compile_options(celerisroute_core PRIVATE /W3 /EHsc /bigobj)
  target_compile_definitions(celerisroute_core PUBLIC NOMINMAX)
else()
  target_compile_options(celerisroute_core PRIVATE -Wall -Wextra)
endif()

# ---------------------------------------------------------------------------
# Python extension module celerisroute._core.
# ---------------------------------------------------------------------------
if(CELERISROUTE_BUILD_PYTHON)
  if(CMAKE_VERSION VERSION_LESS 3.26)
    set(CELERISROUTE_PYTHON_DEV_MODULE Development.Module)
  else()
    set(CELERISROUTE_PYTHON_DEV_MODULE Development.Module OPTIONAL_COMPONENTS Development.SABIModule)
  endif()
  find_package(Python 3.11 REQUIRED COMPONENTS Interpreter ${CELERISROUTE_PYTHON_DEV_MODULE})

  if(NOT nanobind_DIR)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT ERROR_QUIET)
  endif()
  find_package(nanobind CONFIG REQUIRED)

  # FREE_THREADED only has an effect on free-threaded CPython builds, where it
  # declares that the module does not rely on the GIL for its own safety.
  nanobind_add_module(_core FREE_THREADED NB_STATIC LTO cpp/bindings/python.cpp)
  target_link_libraries(_core PRIVATE celerisroute_core)

  if(SKBUILD)
    install(TARGETS _core LIBRARY DESTINATION celerisroute RUNTIME DESTINATION celerisroute)
  endif()
endif()

if(CELERISROUTE_BUILD_BENCHMARKS)
  add_executable(celerisroute_bench cpp/bench/bench_core.cpp)
  target_link_libraries(celerisroute_bench PRIVATE celerisroute_core)
endif()
