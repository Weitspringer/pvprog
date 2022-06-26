# CMakeList.txt: CMake-Projekt für "mcpi". Schließen Sie die Quelle ein, und definieren Sie
# projektspezifische Logik hier.
#
cmake_minimum_required (VERSION 3.8)

# Fügen Sie der ausführbaren Datei dieses Projekts eine Quelle hinzu.
add_executable (mcpi "mcpi.cpp" "mcpi.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET mcpi PROPERTY CXX_STANDARD 20)
endif()

# TODO: Fügen Sie bei Bedarf Tests hinzu, und installieren Sie Ziele.
