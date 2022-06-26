# CMakeList.txt : CMake project for heatmap2, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("heatmap2")

# Add source to this project's executable.
add_executable (heatmap2 "main.cpp" "main.h" "heatmap.cpp"  "heatmap.h" "lifecycle.cpp"  "lifecycle.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET heatmap2 PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
