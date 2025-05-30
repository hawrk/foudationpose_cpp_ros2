// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from dros_common_interfaces:msg/IMUInfo.idl
// generated code does not contain a copyright notice

#ifndef DROS_COMMON_INTERFACES__MSG__DETAIL__IMU_INFO__FUNCTIONS_H_
#define DROS_COMMON_INTERFACES__MSG__DETAIL__IMU_INFO__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "dros_common_interfaces/msg/rosidl_generator_c__visibility_control.h"

#include "dros_common_interfaces/msg/detail/imu_info__struct.h"

/// Initialize msg/IMUInfo message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * dros_common_interfaces__msg__IMUInfo
 * )) before or use
 * dros_common_interfaces__msg__IMUInfo__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_dros_common_interfaces
bool
dros_common_interfaces__msg__IMUInfo__init(dros_common_interfaces__msg__IMUInfo * msg);

/// Finalize msg/IMUInfo message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dros_common_interfaces
void
dros_common_interfaces__msg__IMUInfo__fini(dros_common_interfaces__msg__IMUInfo * msg);

/// Create msg/IMUInfo message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * dros_common_interfaces__msg__IMUInfo__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_dros_common_interfaces
dros_common_interfaces__msg__IMUInfo *
dros_common_interfaces__msg__IMUInfo__create();

/// Destroy msg/IMUInfo message.
/**
 * It calls
 * dros_common_interfaces__msg__IMUInfo__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dros_common_interfaces
void
dros_common_interfaces__msg__IMUInfo__destroy(dros_common_interfaces__msg__IMUInfo * msg);

/// Check for msg/IMUInfo message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_dros_common_interfaces
bool
dros_common_interfaces__msg__IMUInfo__are_equal(const dros_common_interfaces__msg__IMUInfo * lhs, const dros_common_interfaces__msg__IMUInfo * rhs);

/// Copy a msg/IMUInfo message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_dros_common_interfaces
bool
dros_common_interfaces__msg__IMUInfo__copy(
  const dros_common_interfaces__msg__IMUInfo * input,
  dros_common_interfaces__msg__IMUInfo * output);

/// Initialize array of msg/IMUInfo messages.
/**
 * It allocates the memory for the number of elements and calls
 * dros_common_interfaces__msg__IMUInfo__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_dros_common_interfaces
bool
dros_common_interfaces__msg__IMUInfo__Sequence__init(dros_common_interfaces__msg__IMUInfo__Sequence * array, size_t size);

/// Finalize array of msg/IMUInfo messages.
/**
 * It calls
 * dros_common_interfaces__msg__IMUInfo__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dros_common_interfaces
void
dros_common_interfaces__msg__IMUInfo__Sequence__fini(dros_common_interfaces__msg__IMUInfo__Sequence * array);

/// Create array of msg/IMUInfo messages.
/**
 * It allocates the memory for the array and calls
 * dros_common_interfaces__msg__IMUInfo__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_dros_common_interfaces
dros_common_interfaces__msg__IMUInfo__Sequence *
dros_common_interfaces__msg__IMUInfo__Sequence__create(size_t size);

/// Destroy array of msg/IMUInfo messages.
/**
 * It calls
 * dros_common_interfaces__msg__IMUInfo__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_dros_common_interfaces
void
dros_common_interfaces__msg__IMUInfo__Sequence__destroy(dros_common_interfaces__msg__IMUInfo__Sequence * array);

/// Check for msg/IMUInfo message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_dros_common_interfaces
bool
dros_common_interfaces__msg__IMUInfo__Sequence__are_equal(const dros_common_interfaces__msg__IMUInfo__Sequence * lhs, const dros_common_interfaces__msg__IMUInfo__Sequence * rhs);

/// Copy an array of msg/IMUInfo messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_dros_common_interfaces
bool
dros_common_interfaces__msg__IMUInfo__Sequence__copy(
  const dros_common_interfaces__msg__IMUInfo__Sequence * input,
  dros_common_interfaces__msg__IMUInfo__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // DROS_COMMON_INTERFACES__MSG__DETAIL__IMU_INFO__FUNCTIONS_H_
