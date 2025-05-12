cmake_minimum_required(VERSION 3.8)
project(foundationpose_service)

# CUDA 架构设置
cmake_policy(SET CMP0104 NEW)
set(CMAKE_CUDA_ARCHITECTURES 86 87)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)  # 添加 OpenCV
find_package(dros_common_interfaces REQUIRED)
find_package(auto_sam_interfaces REQUIRED)
find_package(glog REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_subdirectory(src/detection_6d_foundationpose)
set(ENABLE_TENSORRT ON)
add_subdirectory(src/easy_deploy_tool)
add_executable(main src/main.cpp)

target_include_directories(main PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(main PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

target_compile_options(main PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20012>
)

ament_target_dependencies(
  main
  rclcpp
  geometry_msgs
  sensor_msgs
  cv_bridge
  OpenCV
  Eigen3
  dros_common_interfaces
  auto_sam_interfaces
)

target_link_libraries(main
  deploy_core
  trt_core
  detection_6d_foundationpose
)


install(TARGETS main
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

ament_package()
