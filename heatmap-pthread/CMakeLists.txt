﻿# CMakeList.txt : CMake project for heatmap, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("heatmap")

# Add source to this project's executable.
add_executable (heatmap "main.cpp" "main.h" "utils.cpp" "utils.h" "heatmap.cpp"  "heatmap.h" "lifecycle.cpp"  "lifecycle.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET heatmap PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
