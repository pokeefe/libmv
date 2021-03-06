# TODO(julien) remove this line, what's that for?
# The Xcode generator on OS X has trouble with this specific string definition
# Should probably still be removed
if(CMAKE_GENERATOR STREQUAL "Xcode")
  ADD_DEFINITIONS(-DTHIS_SOURCE_DIR=\"\\\\"${CMAKE_CURRENT_SOURCE_DIR}\\\\"\")
else()
  ADD_DEFINITIONS(-DTHIS_SOURCE_DIR="\\"${CMAKE_CURRENT_SOURCE_DIR}\\"")
endif()

FIND_PACKAGE( OpenCV REQUIRED )

IF(WIN32)
  INCLUDE_DIRECTORIES(
    ${JPEG_SOURCE_DIR}
    ${PNG_SOURCE_DIR}
    ${ZLIB_SOURCE_DIR}
  )
ENDIF(WIN32)

# define the source files
SET(IMAGE_SRC 
              image_sequence.cc image_sequence_io.cc
)

# define the header files (make the headers appear in IDEs.)
FILE(GLOB IMAGE_HDRS *.h)

ADD_LIBRARY(image ${IMAGE_SRC} ${IMAGE_HDRS})
TARGET_LINK_LIBRARIES(image png jpeg glog gflags ${PTHREAD} ${OpenCV_LIBS} )

# make the name of debug libraries end in _d.
SET_TARGET_PROPERTIES(image PROPERTIES DEBUG_POSTFIX "_d")

# installation rules for the library
LIBMV_INSTALL_LIB(image)

MACRO (IMAGE_TEST NAME)
  LIBMV_TEST(${NAME} image)
ENDMACRO (IMAGE_TEST)

IMAGE_TEST(image_sequence_io)
IMAGE_TEST(lru_cache)
IMAGE_TEST(sample)
IMAGE_TEST(tuple)
