cmake_minimum_required(VERSION 2.8)

project( 3d_eye_tracker )

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
set(CMAKE_INSTALL_PREFIX "./install")


#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

# Your custom paths to Boost and OpenCV libraries
set(OpenCV_DIR "/home/jqzlca/software/opencv3.4.20_install/share/OpenCV")
# Bundled external libraries
#set(CMAKE_PREFIX_PATH "/home/jqzlca/software/boost162_install")
#set(CMAKE_FIND_ROOT_PATH "/home/jqzlca/software/boost162_install")

set(Eigen_INCLUDE_DIR  "/home/jqzlca/software/eigen_install/share/eigen3/cmake")
set(SPII_INSTALL_DIR   "/home/jqzlca/software/spii_install")
set(Eigen3_DIR "/home/jqzlca/software/eigen_install/share/eigen3/cmake")
set(Ceres_DIR  "/home/jqzlca/software/ceres_install/lib/cmake/Ceres")


###################################
### Setup Boost path


set(BOOST_ROOT "/home/jqzlca/software/boost162_install")
set(BOOST_INCLUDEDIR "/home/jqzlca/software/boost162_install/include")
set(BOOST_LIBRARYDIR "/home/jqzlca/software/boost162_install/lib")

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
unset(Boost_INCLUDE_DIR CACHE)
unset(Boost_LIBRARY_DIRS CACHE)
###################################

#find_package ( Boost 1.58 COMPONENTS filesystem system timer REQUIRED )
find_package ( Boost 1.62 EXACT COMPONENTS filesystem system timer serialization REQUIRED)
if(Boost_FOUND)
#    include_directories(${Boost_INCLUDE_DIRS})
    
    MESSAGE( STATUS "Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}.")
    MESSAGE( STATUS "Boost_LIBRARIES = ${Boost_LIBRARIES}.")
    MESSAGE( STATUS "Boost_LIB_VERSION = ${Boost_LIB_VERSION}.")
endif()
find_package ( OpenCV REQUIRED )
#find_package ( TBB REQUIRED )
find_package ( Eigen3 REQUIRED )
find_package (Ceres REQUIRED)
find_package(Threads)
### Find spii
set(spii_INCLUDE_DIRS ${SPII_INSTALL_DIR}/include)
find_library (spii_LIBRARY_RELEASE
              spii
              PATHS ${SPII_INSTALL_DIR}/lib/ ${SPII_INSTALL_DIR}/lib/Release)
find_library (meschach_LIBRARY_RELEASE
              meschach
              PATHS ${SPII_INSTALL_DIR}/lib/ ${SPII_INSTALL_DIR}/lib/Release)
if (spii_LIBRARY_RELEASE AND meschach_LIBRARY_RELEASE)
    set(spii_LIBRARIES ${spii_LIBRARY_RELEASE} ${meschach_LIBRARY_RELEASE})
else()
    set(spii_LIBRARIES "")
endif()
message(STATUS "spii_LIBRARIES:\"${spii_LIBRARIES}\"")
# copy dll
#file(COPY ${SPII_INSTALL_DIR}/bin/spii.dll DESTINATION ${PROJECT_BINARY_DIR}/main)

# copy OpenCV dlls
file(GLOB OPENCV_DLLS
  "${OpenCV_DIR}/x64/vc14/bin/*.dll"
)
file(COPY ${OPENCV_DLLS} DESTINATION ${PROJECT_BINARY_DIR}/main)

message(STATUS "")
message(STATUS "--- Include directories ---" )
message(STATUS " OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}" )
message(STATUS " Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}" )
message(STATUS "---------------------------" )
message(STATUS "")

add_subdirectory (main)
add_subdirectory (singleeyefitter)
