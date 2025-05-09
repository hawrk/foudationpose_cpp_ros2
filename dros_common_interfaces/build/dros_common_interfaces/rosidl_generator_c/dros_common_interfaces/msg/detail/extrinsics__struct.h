// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dros_common_interfaces:msg/Extrinsics.idl
// generated code does not contain a copyright notice

#ifndef DROS_COMMON_INTERFACES__MSG__DETAIL__EXTRINSICS__STRUCT_H_
#define DROS_COMMON_INTERFACES__MSG__DETAIL__EXTRINSICS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/Extrinsics in the package dros_common_interfaces.
/**
  * Cross-stream extrinsics: encodes the topology describing how the different devices are oriented
 */
typedef struct dros_common_interfaces__msg__Extrinsics
{
  /// Column - major 3x3 rotation matrix
  double rotation[9];
  /// Three-element translation vector, in meters
  double translation[3];
} dros_common_interfaces__msg__Extrinsics;

// Struct for a sequence of dros_common_interfaces__msg__Extrinsics.
typedef struct dros_common_interfaces__msg__Extrinsics__Sequence
{
  dros_common_interfaces__msg__Extrinsics * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dros_common_interfaces__msg__Extrinsics__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DROS_COMMON_INTERFACES__MSG__DETAIL__EXTRINSICS__STRUCT_H_
