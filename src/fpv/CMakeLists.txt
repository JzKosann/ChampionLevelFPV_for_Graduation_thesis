cmake_minimum_required(VERSION 3.26)
project(fpv)
enable_language(CUDA)
enable_language(CXX)

set(CMAKE_CUDA_STANDARD 20)

set(cv_bridge_DIR /usr/local/cv_bridge/share/cv_bridge/cmake)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        geometry_msgs
        std_msgs
        image_transport
        cv_bridge
#        camera_models
#        tf
)
find_package(CUDAToolkit REQUIRED)
#/usr/local/cuda-11.4/bin/nvcc
include("~/Documents/opencv-4.6.0/build/OpenCVConfig.cmake")
#set(OpenCV_DIR /usr/local/opencv3.4.14/share/OpenCV)
find_package(Eigen3)
#find_package(OpenCV 4.6.0 REQUIRED)

include_directories(
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
)

catkin_package()

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "    catkin libraries: ${catkin_LIBRARIES}")

if (CMAKE_CUDA_COMPILER)
    message(STATUS "nvcc path : ${CMAKE_CUDA_COMPILER}")
else ()
    message(WARNING "nvcc not found. Please check CUDA is installed correctly!")
endif ()
###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
        ${catkin_INCLUDE_DIRS}
        include/fpv
        ${PROJECT_SOURCE_DIR}/include
)

file(GLOB_RECURSE SOURCES "src/*.*" "include/*.*")
add_executable(fpv_node src/rosNode.cpp ${SOURCES}
)

set_target_properties(
        fpv_node
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(fpv_node
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        ${CUDA_cudart_LIBRARY}
)
