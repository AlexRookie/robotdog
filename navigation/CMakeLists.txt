cmake_minimum_required(VERSION 3.0)
project(robodog_nav)

set(CMAKE_CXX_STANDARD 17)

# openCV
find_package(OpenCV REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/zmq)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Clothoids/lib/include)
include_directories(/usr/include/eigen3)
include_directories(${OPENCV_INCLUDE_DIRS})

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Clothoids/lib/lib)
link_directories(${OpenCV_LIBRARY_DIRS})

#set(CMAKE_BUILD_TYPE Debug)
add_compile_options(-Wno-ignored-attributes)

################### ROBODOG_NAVIGATION ###############################
ADD_EXECUTABLE(navigation
  source/zmq/Publisher.cc
  source/zmq/Subscriber.cc
  source/zmq/RequesterSimple.cpp
  source/control.cpp
  source/hardwareglobalinterface.cpp
  source/pathFitter.cpp
  source/robodogController.cpp
  source/smooth.cpp
  source/main.cpp
)



target_link_libraries(navigation
  pthread
  zmq
  Clothoids_linux
  ${OpenCV_LIBRARIES}
)


