# define the source files
SET(BASE_SRC vector_test.cc)

# define the header files (make the headers appear in IDEs.)
FILE(GLOB BASE_HDRS *.h)

ADD_LIBRARY(base ${BASE_SRC} ${BASE_HDRS})
TARGET_LINK_LIBRARIES(base gtest gflags glog)

LIBMV_TEST(vector numeric)
