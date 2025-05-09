// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dros_common_interfaces:msg/RGBD.idl
// generated code does not contain a copyright notice

#ifndef DROS_COMMON_INTERFACES__MSG__DETAIL__RGBD__STRUCT_H_
#define DROS_COMMON_INTERFACES__MSG__DETAIL__RGBD__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'rgb_camera_info'
// Member 'depth_camera_info'
#include "sensor_msgs/msg/detail/camera_info__struct.h"
// Member 'rgb'
// Member 'depth'
#include "sensor_msgs/msg/detail/image__struct.h"

/// Struct defined in msg/RGBD in the package dros_common_interfaces.
/**
  * RGBD Message
 */
typedef struct dros_common_interfaces__msg__RGBD
{
  std_msgs__msg__Header header;
  sensor_msgs__msg__CameraInfo rgb_camera_info;
  sensor_msgs__msg__CameraInfo depth_camera_info;
  sensor_msgs__msg__Image rgb;
  sensor_msgs__msg__Image depth;
} dros_common_interfaces__msg__RGBD;

// Struct for a sequence of dros_common_interfaces__msg__RGBD.
typedef struct dros_common_interfaces__msg__RGBD__Sequence
{
  dros_common_interfaces__msg__RGBD * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dros_common_interfaces__msg__RGBD__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DROS_COMMON_INTERFACES__MSG__DETAIL__RGBD__STRUCT_H_
