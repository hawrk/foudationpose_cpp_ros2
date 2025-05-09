// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__rosidl_typesupport_fastrtps_cpp.hpp.em
// with input from dros_common_interfaces:msg/RGBD.idl
// generated code does not contain a copyright notice

#ifndef DROS_COMMON_INTERFACES__MSG__DETAIL__RGBD__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
#define DROS_COMMON_INTERFACES__MSG__DETAIL__RGBD__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_

#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "dros_common_interfaces/msg/rosidl_typesupport_fastrtps_cpp__visibility_control.h"
#include "dros_common_interfaces/msg/detail/rgbd__struct.hpp"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

#include "fastcdr/Cdr.h"

namespace dros_common_interfaces
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_dros_common_interfaces
cdr_serialize(
  const dros_common_interfaces::msg::RGBD & ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_dros_common_interfaces
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  dros_common_interfaces::msg::RGBD & ros_message);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_dros_common_interfaces
get_serialized_size(
  const dros_common_interfaces::msg::RGBD & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_dros_common_interfaces
max_serialized_size_RGBD(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace dros_common_interfaces

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_dros_common_interfaces
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, dros_common_interfaces, msg, RGBD)();

#ifdef __cplusplus
}
#endif

#endif  // DROS_COMMON_INTERFACES__MSG__DETAIL__RGBD__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
