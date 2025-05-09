cmake_minimum_required(VERSION 3.8)
project(data_collection_service)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
option(BUILD_WITH_OPENMP "Use OpenMP" OFF)
option(SET_USER_BREAK_AT_STARTUP "Set user wait point in startup (for debug)" OFF)

set(CMAKE_CXX_STANDARD 17)

# Compiler Defense Flags
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -z noexecstack -z relro -z now")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -z noexecstack -z relro -z now")
# Compiler flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIE -fstack-protector-strong")
# Generic flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fno-operator-names -Wformat -Wformat-security -Wall")
# Dot not forward c++ flag to GPU beucause it is not supported
set(CUDA_PROPAGATE_HOST_FLAGS OFF )
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D_FORTIFY_SOURCE=2")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie")

if(BUILD_WITH_OPENMP)
    find_package(OpenMP)
    if(NOT OpenMP_FOUND)
        message(FATAL_ERROR "\n\n OpenMP is missing!\n\n")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fopenmp")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

if(SET_USER_BREAK_AT_STARTUP)
    message("GOT FLAG IN CmakeLists.txt")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBPDEBUG")
endif()


#导入 realsense SDK2.0   TODO: 上线需修改路径
#set(REALSENSE_PATH "/home/byd/third_party/real_sense")

#set(CMAKE_PREFIX_PATH ${REALSENSE_PATH}/lib/cmake/realsense2)

# realsense2 library
#include_directories(${REALSENSE_PATH}/include)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/wrapper/realsense/include)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(diagnostic_updater REQUIRED)

find_package(dros_common_interfaces REQUIRED)

find_package(OpenCV REQUIRED COMPONENTS core)

find_package(realsense2 2.55.1)

if (BUILD_ACCELERATE_GPU_WITH_GLSL)
  find_package(realsense2-gl 2.55.1)
endif()

if(NOT realsense2_FOUND)
    message(FATAL_ERROR "\n\n Intel RealSense SDK 2.0 is missing, please install it from https://github.com/IntelRealSense/librealsense/releases\n\n")
endif()

include_directories(${OpenCV_INCLUDE_DIRS})  # add OpenCV includes to the included dirs

set(REALSENSE_PREFIX "wrapper/realsense")

#[[set(SOURCES
    ${REALSENSE_PREFIX}/src/realsense_node_factory.cpp
    ${REALSENSE_PREFIX}/src/base_realsense_node.cpp
    ${REALSENSE_PREFIX}/src/base_parameters.cpp
    ${REALSENSE_PREFIX}/src/base_node_setup.cpp
    ${REALSENSE_PREFIX}/src/ros_sensor.cpp
    ${REALSENSE_PREFIX}/src/ros_utils.cpp
    ${REALSENSE_PREFIX}/src/dynamic_params.cpp
    ${REALSENSE_PREFIX}/src/sensor_params.cpp
    ${REALSENSE_PREFIX}/src/named_filter.cpp
    ${REALSENSE_PREFIX}/src/profile_manager.cpp
    ${REALSENSE_PREFIX}/src/image_publisher.cpp
    ${REALSENSE_PREFIX}/src/base_tfs.cpp
    ${REALSENSE_PREFIX}/src/ros_param_backend.cpp
  )
    ]]
#or
file(GLOB SOURCES
  "${REALSENSE_PREFIX}/src/*.cpp")

if (BUILD_ACCELERATE_GPU_WITH_GLSL)
  list(APPEND SOURCES ${REALSENSE_PREFIX}/src/gl_gpu_processing.cpp)
endif()

# The header 'cv_bridge/cv_bridge.hpp' was added in version 3.3.0. For older
# cv_bridge versions, we have to use the header 'cv_bridge/cv_bridge.h'.
if(${cv_bridge_VERSION} VERSION_GREATER_EQUAL "3.3.0")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCV_BRDIGE_HAS_HPP")
endif()

# 'OnSetParametersCallbackType' is only defined for rclcpp 17 and onward.
if(${rclcpp_VERSION} VERSION_GREATER_EQUAL "17.0")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DRCLCPP_HAS_OnSetParametersCallbackType")
endif()

if (BUILD_ACCELERATE_GPU_WITH_GLSL)
  add_definitions(-DACCELERATE_GPU_WITH_GLSL)
endif()

#[[set(INCLUDES
    ${REALSENSE_PREFIX}/include/constants.h
    ${REALSENSE_PREFIX}/include/realsense_node_factory.h
    ${REALSENSE_PREFIX}/include/base_realsense_node.h
    ${REALSENSE_PREFIX}/include/ros_sensor.h
    ${REALSENSE_PREFIX}/include/ros_utils.h
    ${REALSENSE_PREFIX}/include/dynamic_params.h
    ${REALSENSE_PREFIX}/include/sensor_params.h
    ${REALSENSE_PREFIX}/include/named_filter.h
    ${REALSENSE_PREFIX}/include/ros_param_backend.h
    ${REALSENSE_PREFIX}/include/profile_manager.h
    ${REALSENSE_PREFIX}/include/image_publisher.h)
    ]]

#or
file(GLOB INCLUDES
  "${REALSENSE_PREFIX}/include/*.h")

if (BUILD_ACCELERATE_GPU_WITH_GLSL)
  list(APPEND INCLUDES ${REALSENSE_PREFIX}/include/gl_window.h)
endif()

add_library(${PROJECT_NAME} SHARED
    ${INCLUDES}
    ${SOURCES}
)

if (BUILD_ACCELERATE_GPU_WITH_GLSL)
  set(link_libraries ${realsense2-gl_LIBRARY})
else()
  set(link_libraries ${realsense2_LIBRARY})
endif()

list(APPEND link_libraries ${OpenCV_LIBS})  # add OpenCV libs to link_libraries

target_link_libraries(${PROJECT_NAME}
    ${link_libraries}
)

set(dependencies
  cv_bridge
  image_transport
  rclcpp
  rclcpp_components
  dros_common_interfaces
  std_msgs
  sensor_msgs
  nav_msgs
  tf2
  tf2_ros
  diagnostic_updater
)

if (BUILD_ACCELERATE_GPU_WITH_GLSL)
  list(APPEND dependencies realsense2-gl)
else()
  list(APPEND dependencies realsense2)
endif()

ament_target_dependencies(${PROJECT_NAME}
  ${dependencies}
)

#生成插件
rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "realsense2_camera::RealSenseNodeFactory"
  EXECUTABLE realsense2_camera_node
)

# Install binaries
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

#安装realsense2 SDK
#install(DIRECTORY ${REALSENSE_PATH}/lib/
#  DESTINATION lib
#)


# Install headers
install(
  DIRECTORY ${REALSENSE_PREFIX}/include/
  DESTINATION include
)

# Install launch files
install(DIRECTORY 
    launch
    DESTINATION share/${PROJECT_NAME}
    )

#单独执行体
add_executable(manager src/main.cpp src/data_collection_manager.cpp)

target_include_directories(manager PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(manager PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

ament_target_dependencies(
  manager
  "rclcpp"
  "std_msgs"
)

install(TARGETS manager
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Ament exports
ament_export_include_directories(${REALSENSE_PREFIX}/include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(${dependencies})

ament_package()
