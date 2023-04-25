cmake_minimum_required(VERSION 3.0)
project(robot_dog)

set(CMAKE_CXX_STANDARD 14)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
include_directories(${OPENCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Clothoids/lib/include)

# openCV
find_package(OpenCV REQUIRED)
link_directories(${OpenCV_LIBRARY_DIRS})

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Clothoids/lib/lib)

include_directories(/usr/include/eigen3)

#SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall")

################### FUSION ###################
ADD_EXECUTABLE(fusion
  source/mainFusion.cpp
  source/zmq/Publisher.cc
  source/zmq/Subscriber.cc
  source/params.cpp
)

target_link_libraries(fusion
  pthread
  zmq
)

################### FILTER ###################
ADD_EXECUTABLE(filter
  source/mainFilter.cpp
  source/zmq/Publisher.cc
  source/zmq/Subscriber.cc
)

target_link_libraries(filter
  pthread
  zmq
  ${OpenCV_LIBRARIES}
)

################### VISUAL DEBUG ###################
ADD_EXECUTABLE(vdebug
  source/visualDebug.cpp
  source/MapManager.cpp
  source/zmq/Publisher.cc
  source/zmq/Subscriber.cc
  source/params.cpp
)

target_link_libraries(vdebug
  pthread
  zmq
  ${OpenCV_LIBRARIES}
  yaml-cpp stdc++fs
  Clothoids_linux
)
