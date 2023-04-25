# usefull guide to cmake: https://www.youtube.com/playlist?list=PLK6MXr8gasrGmIiSuVQXpfFuE1uPT615s
cmake_minimum_required(VERSION 3.10)

### Configuration
enable_language(CXX) 				#set the language as c++
set (CMAKE_CXX_STANDARD 17)			#set to c++17 to allow the use of optionals and unique_ptr... (if not c++11 is enough)
set (CMAKE_CXX_STANDARD_REQUIRED TRUE)
set (CMAKE_CXX_EXTENSIONS FALSE)
project(CMake_template VERSION 1.2)	#set project name (useful for knowledge)
if (UNIX)
    set(CMAKE_CXX_FLAGS "-O3")		#linux optimization
endif (UNIX)
if (WIN32)							#windows optimization		
endif (WIN32)


set(CMAKE_BUILD_TYPE Release)

find_package(realsense2 REQUIRED )
include_directories(${realsense_INCLUDE_DIR})
add_definitions(-DREALSENSE_3D)

SET(NO_CUDA FALSE)
if (${NO_CUDA}) 
    add_definitions(-DNO_CUDA)
    message(WARNING "CUDA disabled by CMakeLists.txt")
endif()


### Create alias
include_directories(${PROJECT_SOURCE_DIR}/src/include)            	#include the .hh folder, then each .cc will load its .hh
file(GLOB source_files                                              #include all the .cc files as source
    "${PROJECT_SOURCE_DIR}/src/*.cc" 
    "${PROJECT_SOURCE_DIR}/src/zmq/*.cc") 

if (${NO_CUDA})
    list(REMOVE_ITEM source_files ${PROJECT_SOURCE_DIR}/src/trtNet.cc) # remove TensorRT dependencies if CUDA not enabled
else()
    find_package(CUDA REQUIRED)

    set(TENSOR_RT_INC /usr/include/aarch64-linux-gnu)
    set(TENSOR_RT_LIB /usr/lib/aarch64-linux-gnu)

    INCLUDE_DIRECTORIES(
        /usr/local/cuda-10.2/include
        ${TENSOR_RT_INC}
    )
    LINK_DIRECTORIES(
      /usr/local/cuda-10.2/lib64
    )
endif()


### Requirements
#find_package(OpenCV REQUIRED)					#importing of openCV (it need to be installed on the machine)
#include_directories(${OpenCV_INCLUDE_DIRS})		#include all source files of openCV needed to compile
find_package(PkgConfig REQUIRED) 
pkg_check_modules(OpenCV REQUIRED opencv4)
include_directories(${OpenCV_INCLUDE_DIRS})		#include all source files of openCV needed to compile
message(STATUS ${OpenCV_INCLUDE_DIRS})
message(STATUS ${OpenCV_LDFLAGS})



## Create the Static library
set(LIB  Biblioteca)						#set the name of the static library (.a) as projectLib.a into the build directory
add_library(${LIB} STATIC ${source_files})	#add all the .cc to static library previously defined


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
### Create the executables
#####################   TEST SSD and YOLO   ################################################
# file(GLOB source_detectorSsdYolo   "${PROJECT_SOURCE_DIR}/exec/detector_SsdYolo.cc")	#include the main
# set(exec_detectorSsdYolo detectorSsdYolo)												#set the name of the executable
# add_executable(${exec_detectorSsdYolo} ${source_detectorSsdYolo})						#declare the executable with its file
# target_link_libraries( ${exec_detectorSsdYolo} ${LIB} ${OpenCV_LIBS})					#add the library to the created executable	

# file(GLOB source_onlineYOLO   "${PROJECT_SOURCE_DIR}/exec/detector_onlineYOLO.cc")
# set(exec_onlineYOLO detectorOnlineYOLO)
# add_executable(${exec_onlineYOLO} ${source_onlineYOLO})
# target_link_libraries( ${exec_onlineYOLO} ${OpenCV_LIBS})

# file(GLOB source_onlineSSD   "${PROJECT_SOURCE_DIR}/exec/detector_onlineSSD.cc")
# set(exec_onlineSSD detectorOnlineSSD)
# add_executable(${exec_onlineSSD} ${source_onlineSSD})
# target_link_libraries( ${exec_onlineSSD} ${OpenCV_LIBS})

# #####################   TEST GoogleNet and ResNet50   ######################################
# file(GLOB source_classifierGoogleResNet   "${PROJECT_SOURCE_DIR}/exec/classifier_GoogleResNet.cc")
# set(exec_classifierGoogleResNet classifierGoogleResNet)
# add_executable(${exec_classifierGoogleResNet} ${source_classifierGoogleResNet})
# target_link_libraries( ${exec_classifierGoogleResNet} ${LIB} ${OpenCV_LIBS})

# #####################   TEST DNN MODELS   ##################################################
# file(GLOB source_dnnModel   "${PROJECT_SOURCE_DIR}/exec/test_dnnModel.cc")
# set(exec_dnnModel dnnModel)
# add_executable(${exec_dnnModel} ${source_dnnModel})
# target_link_libraries( ${exec_dnnModel} ${LIB} ${OpenCV_LIBS})

# #####################   TEST openCV trackers   #############################################
# file(GLOB source_trackerAll   "${PROJECT_SOURCE_DIR}/exec/tracker_all.cc")
# set(exec_trackerAll trackerAll)
# add_executable(${exec_trackerAll} ${source_trackerAll})
# target_link_libraries( ${exec_trackerAll} ${LIB} ${OpenCV_LIBS})

# #####################   TEST KNN   #########################################################
# file(GLOB source_knn   "${PROJECT_SOURCE_DIR}/exec/test_knn.cc")
# set(exec_knn knn)
# add_executable(${exec_knn} ${source_knn})
# target_link_libraries( ${exec_knn} ${LIB} ${OpenCV_LIBS})

# #####################   MAIN of FOLLOWER   #################################################
file(GLOB source_main   "${PROJECT_SOURCE_DIR}/exec/main3D.cc")
set(exec_main main)

if (${NO_CUDA})
    add_executable(${exec_main} ${source_main})
    target_link_libraries( ${exec_main} ${LIB} ${OpenCV_LDFLAGS}  ${realsense2_LIBRARY} pthread zmq) 
else()
    cuda_add_executable(${exec_main} ${source_main})
    target_link_libraries( ${exec_main} ${LIB} ${OpenCV_LDFLAGS} ${realsense2_LIBRARY} pthread zmq cuda cudart  ${TENSOR_RT_LIB}/libnvcaffe_parser.so ${TENSOR_RT_LIB}/libnvinfer.so) #tutto OK
endif()






