cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(superpoint)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(superpoint src/sp.cpp)
target_link_libraries(superpoint "${TORCH_LIBRARIES}")
target_link_libraries(superpoint ${OpenCV_LIBS})
set_property(TARGET superpoint PROPERTY CXX_STANDARD 17)

#set_property(TARGET superpoint PROPERTY CXX_STANDARD 11)
