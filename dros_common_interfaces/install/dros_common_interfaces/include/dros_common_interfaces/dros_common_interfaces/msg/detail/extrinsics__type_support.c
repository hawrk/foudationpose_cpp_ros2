// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from dros_common_interfaces:msg/Extrinsics.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "dros_common_interfaces/msg/detail/extrinsics__rosidl_typesupport_introspection_c.h"
#include "dros_common_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "dros_common_interfaces/msg/detail/extrinsics__functions.h"
#include "dros_common_interfaces/msg/detail/extrinsics__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__Extrinsics_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  dros_common_interfaces__msg__Extrinsics__init(message_memory);
}

void dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__Extrinsics_fini_function(void * message_memory)
{
  dros_common_interfaces__msg__Extrinsics__fini(message_memory);
}

size_t dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__size_function__Extrinsics__rotation(
  const void * untyped_member)
{
  (void)untyped_member;
  return 9;
}

const void * dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__get_const_function__Extrinsics__rotation(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__get_function__Extrinsics__rotation(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__fetch_function__Extrinsics__rotation(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__get_const_function__Extrinsics__rotation(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__assign_function__Extrinsics__rotation(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__get_function__Extrinsics__rotation(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__size_function__Extrinsics__translation(
  const void * untyped_member)
{
  (void)untyped_member;
  return 3;
}

const void * dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__get_const_function__Extrinsics__translation(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__get_function__Extrinsics__translation(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__fetch_function__Extrinsics__translation(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__get_const_function__Extrinsics__translation(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__assign_function__Extrinsics__translation(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__get_function__Extrinsics__translation(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

static rosidl_typesupport_introspection_c__MessageMember dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__Extrinsics_message_member_array[2] = {
  {
    "rotation",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    9,  // array size
    false,  // is upper bound
    offsetof(dros_common_interfaces__msg__Extrinsics, rotation),  // bytes offset in struct
    NULL,  // default value
    dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__size_function__Extrinsics__rotation,  // size() function pointer
    dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__get_const_function__Extrinsics__rotation,  // get_const(index) function pointer
    dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__get_function__Extrinsics__rotation,  // get(index) function pointer
    dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__fetch_function__Extrinsics__rotation,  // fetch(index, &value) function pointer
    dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__assign_function__Extrinsics__rotation,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "translation",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    3,  // array size
    false,  // is upper bound
    offsetof(dros_common_interfaces__msg__Extrinsics, translation),  // bytes offset in struct
    NULL,  // default value
    dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__size_function__Extrinsics__translation,  // size() function pointer
    dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__get_const_function__Extrinsics__translation,  // get_const(index) function pointer
    dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__get_function__Extrinsics__translation,  // get(index) function pointer
    dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__fetch_function__Extrinsics__translation,  // fetch(index, &value) function pointer
    dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__assign_function__Extrinsics__translation,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__Extrinsics_message_members = {
  "dros_common_interfaces__msg",  // message namespace
  "Extrinsics",  // message name
  2,  // number of fields
  sizeof(dros_common_interfaces__msg__Extrinsics),
  dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__Extrinsics_message_member_array,  // message members
  dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__Extrinsics_init_function,  // function to initialize message memory (memory has to be allocated)
  dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__Extrinsics_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__Extrinsics_message_type_support_handle = {
  0,
  &dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__Extrinsics_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_dros_common_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, dros_common_interfaces, msg, Extrinsics)() {
  if (!dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__Extrinsics_message_type_support_handle.typesupport_identifier) {
    dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__Extrinsics_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &dros_common_interfaces__msg__Extrinsics__rosidl_typesupport_introspection_c__Extrinsics_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
