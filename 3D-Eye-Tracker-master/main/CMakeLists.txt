

include_directories (
#  "${PROJECT_SOURCE_DIR}"
  #"../external/DirectShowFrameGrabber"
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${spii_INCLUDE_DIRS}
  "../singleeyefitter"
)


file (GLOB SRCS
  "*.cpp"
  "./pupil/*.cpp"
  #"../external/DirectShowFrameGrabber/*.cpp"
  #"../external/DirectShowFrameGrabber/*.c"
  # "external/DirectShowFrameGrabber/*.idl"pupillabs
  )

file (GLOB HEADERS
  "*.h"
  "*.hpp"
  "./pupil/*.h"
  #"../external/DirectShowFrameGrabber/*.h"
  )
include_directories(include pupil)

add_executable( main ${SRCS} ${HEADERS} )
target_link_libraries( main 
		       ${OpenCV_LIBS}
		       ${Boost_FILESYSTEM_LIBRARY}
		       ${Boost_SYSTEM_LIBRARY}
		       ${Boost_LIBRARIES}
           ${spii_LIBRARIES}
             ${CMAKE_THREAD_LIBS_INIT}
           "singleeyefitter"
           )
