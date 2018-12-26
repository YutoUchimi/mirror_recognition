cmake_minimum_required(VERSION 2.8.3)
project(mirror_recognition)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif()

find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
  jsk_recognition_msgs
  jsk_recognition_utils
  message_filters
  nodelet
  roscpp
  sensor_msgs
)

find_package(Boost REQUIRED COMPONENTS thread)

find_package(Eigen3)
if(EIGEN3_FOUND)
  set(Eigen_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
else()
  find_package(Eigen REQUIRED)
  set(Eigen_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS})
endif()

find_package(PCL REQUIRED COMPONENTS filters)

generate_dynamic_reconfigure_options(
  cfg/PublishRawDataset.cfg
)

catkin_package(
  INCLUDE_DIRS
    include
    ${Eigen_INCLUDE_DIRS}
  LIBRARIES mirror_recognition
  CATKIN_DEPENDS
    jsk_recognition_msgs
    jsk_recognition_utils
    message_filters
    nodelet
    roscpp
    sensor_msgs
  DEPENDS
    Boost
    PCL
)

include_directories(
  include
  ${Boost_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/point_cloud_mirror_flipper_nodelet.cpp
)
target_link_libraries(${PROJECT_NAME}
  ${Boost_LIBRARIES}
  ${catkin_LIBRARIES}
  ${Eigen_LIBRARIES}
  ${PCL_LIBRARIES}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(
  TARGETS
    ${PROJECT_NAME}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(
  DIRECTORY
    learning_scripts
    node_scripts
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  USE_SOURCE_PERMISSIONS
)

install(
  DIRECTORY
    config
    datasets
    launch
    models
    plugins
    rvizconfig
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
