find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

ADD_SUBDIRECTORY(base)
ADD_SUBDIRECTORY(camera)
ADD_SUBDIRECTORY(correspondence)
ADD_SUBDIRECTORY(image)
ADD_SUBDIRECTORY(multiview)
ADD_SUBDIRECTORY(numeric)
ADD_SUBDIRECTORY(reconstruction)
ADD_SUBDIRECTORY(scene_graph)
ADD_SUBDIRECTORY(simple_pipeline)
#ADD_SUBDIRECTORY(tools)
ADD_SUBDIRECTORY(tracking)

LIBMV_INSTALL_ALL_HEADERS()
