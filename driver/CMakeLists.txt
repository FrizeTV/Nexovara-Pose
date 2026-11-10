# Nexovara Pose SteamVR driver (driver_nexovarapose). Builds on Windows (Visual Studio 2019/2022)
# and Linux (gcc/clang). Only needs the bundled openvr_driver.h - no downloads.
cmake_minimum_required(VERSION 3.16)
project(nexovarapose_driver CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(NEXOVARAPOSE_BUILD_TESTS "Build the protocol unit test" ON)
set(NEXOVARAPOSE_DIST "${CMAKE_CURRENT_SOURCE_DIR}/dist" CACHE PATH "Where the ready driver folder goes")

if(WIN32)
  set(ARCH_DIR win64)
  # static C++ runtime: no Visual C++ redistributable needed on the user's PC
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
  set(ARCH_DIR linux64)
endif()

set(OUT_ROOT "${NEXOVARAPOSE_DIST}/nexovarapose")

add_library(driver_nexovarapose SHARED
  src/provider.cpp
  src/devices.cpp
  src/receiver.cpp
  third_party/valve/hand_simulation.cpp)
target_include_directories(driver_nexovarapose PRIVATE src)
# third-party headers: SYSTEM -> their warnings do not hide ours
target_include_directories(driver_nexovarapose SYSTEM PRIVATE third_party/openvr third_party/valve)
set_target_properties(driver_nexovarapose PROPERTIES
  PREFIX ""
  CXX_VISIBILITY_PRESET hidden
  RUNTIME_OUTPUT_DIRECTORY "$<1:${OUT_ROOT}/bin/${ARCH_DIR}>"
  LIBRARY_OUTPUT_DIRECTORY "$<1:${OUT_ROOT}/bin/${ARCH_DIR}>")
if(MSVC)
  target_compile_options(driver_nexovarapose PRIVATE /W4 /O2 /permissive- /utf-8)
  target_compile_definitions(driver_nexovarapose PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
  target_link_libraries(driver_nexovarapose PRIVATE ws2_32)
else()
  target_compile_options(driver_nexovarapose PRIVATE -Wall -Wextra -O2)
  target_link_libraries(driver_nexovarapose PRIVATE pthread)
endif()

# Valve sample code: build it, but don't drown our warnings in theirs
set_source_files_properties(third_party/valve/hand_simulation.cpp PROPERTIES
  COMPILE_OPTIONS "$<IF:$<CXX_COMPILER_ID:MSVC>,/w,-w>")

# copy manifest + resources next to the binary -> a complete SteamVR driver folder
add_custom_command(TARGET driver_nexovarapose PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/nexovarapose" "${OUT_ROOT}"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/third_party/openvr/LICENSE"
          "${OUT_ROOT}/LICENSE-openvr.txt")

if(NEXOVARAPOSE_BUILD_TESTS)
  enable_testing()
  add_executable(nexovarapose_protocol_test tests/test_protocol.cpp src/receiver.cpp)
  target_include_directories(nexovarapose_protocol_test PRIVATE src)
  if(WIN32)
    target_link_libraries(nexovarapose_protocol_test PRIVATE ws2_32)
  else()
    target_link_libraries(nexovarapose_protocol_test PRIVATE pthread)
  endif()
  add_test(NAME protocol COMMAND nexovarapose_protocol_test)
  # mock vrserver that loads the real driver library
  add_executable(nexovarapose_driver_host_test tests/test_driver_host.cpp)
  target_include_directories(nexovarapose_driver_host_test SYSTEM PRIVATE third_party/openvr)
  if(WIN32)
    target_link_libraries(nexovarapose_driver_host_test PRIVATE ws2_32)
  else()
    target_link_libraries(nexovarapose_driver_host_test PRIVATE dl pthread)
  endif()
  add_dependencies(nexovarapose_driver_host_test driver_nexovarapose)
  add_test(NAME driver_host COMMAND nexovarapose_driver_host_test $<TARGET_FILE:driver_nexovarapose>)
endif()
