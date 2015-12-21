cmake_minimum_required(VERSION 3.3)
project(VisionAssignment3)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp Utils.cpp Utils.hpp MedianBackground.cpp MedianBackground.hpp)
add_executable(VisionAssignment3 ${SOURCE_FILES})
target_link_libraries( VisionAssignment3 ${OpenCV_LIBS})