// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from dros_common_interfaces:srv/DeviceInfo.idl
// generated code does not contain a copyright notice

#ifndef DROS_COMMON_INTERFACES__SRV__DETAIL__DEVICE_INFO__STRUCT_H_
#define DROS_COMMON_INTERFACES__SRV__DETAIL__DEVICE_INFO__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in srv/DeviceInfo in the package dros_common_interfaces.
typedef struct dros_common_interfaces__srv__DeviceInfo_Request
{
  uint8_t structure_needs_at_least_one_member;
} dros_common_interfaces__srv__DeviceInfo_Request;

// Struct for a sequence of dros_common_interfaces__srv__DeviceInfo_Request.
typedef struct dros_common_interfaces__srv__DeviceInfo_Request__Sequence
{
  dros_common_interfaces__srv__DeviceInfo_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dros_common_interfaces__srv__DeviceInfo_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'device_name'
// Member 'serial_number'
// Member 'firmware_version'
// Member 'usb_type_descriptor'
// Member 'firmware_update_id'
// Member 'sensors'
// Member 'physical_port'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/DeviceInfo in the package dros_common_interfaces.
typedef struct dros_common_interfaces__srv__DeviceInfo_Response
{
  rosidl_runtime_c__String device_name;
  rosidl_runtime_c__String serial_number;
  rosidl_runtime_c__String firmware_version;
  rosidl_runtime_c__String usb_type_descriptor;
  rosidl_runtime_c__String firmware_update_id;
  rosidl_runtime_c__String sensors;
  rosidl_runtime_c__String physical_port;
} dros_common_interfaces__srv__DeviceInfo_Response;

// Struct for a sequence of dros_common_interfaces__srv__DeviceInfo_Response.
typedef struct dros_common_interfaces__srv__DeviceInfo_Response__Sequence
{
  dros_common_interfaces__srv__DeviceInfo_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} dros_common_interfaces__srv__DeviceInfo_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // DROS_COMMON_INTERFACES__SRV__DETAIL__DEVICE_INFO__STRUCT_H_
