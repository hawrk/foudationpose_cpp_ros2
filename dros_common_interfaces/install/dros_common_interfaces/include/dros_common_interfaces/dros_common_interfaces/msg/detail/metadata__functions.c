// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from dros_common_interfaces:msg/Metadata.idl
// generated code does not contain a copyright notice
#include "dros_common_interfaces/msg/detail/metadata__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `json_data`
#include "rosidl_runtime_c/string_functions.h"

bool
dros_common_interfaces__msg__Metadata__init(dros_common_interfaces__msg__Metadata * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    dros_common_interfaces__msg__Metadata__fini(msg);
    return false;
  }
  // json_data
  if (!rosidl_runtime_c__String__init(&msg->json_data)) {
    dros_common_interfaces__msg__Metadata__fini(msg);
    return false;
  }
  return true;
}

void
dros_common_interfaces__msg__Metadata__fini(dros_common_interfaces__msg__Metadata * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // json_data
  rosidl_runtime_c__String__fini(&msg->json_data);
}

bool
dros_common_interfaces__msg__Metadata__are_equal(const dros_common_interfaces__msg__Metadata * lhs, const dros_common_interfaces__msg__Metadata * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // json_data
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->json_data), &(rhs->json_data)))
  {
    return false;
  }
  return true;
}

bool
dros_common_interfaces__msg__Metadata__copy(
  const dros_common_interfaces__msg__Metadata * input,
  dros_common_interfaces__msg__Metadata * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // json_data
  if (!rosidl_runtime_c__String__copy(
      &(input->json_data), &(output->json_data)))
  {
    return false;
  }
  return true;
}

dros_common_interfaces__msg__Metadata *
dros_common_interfaces__msg__Metadata__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dros_common_interfaces__msg__Metadata * msg = (dros_common_interfaces__msg__Metadata *)allocator.allocate(sizeof(dros_common_interfaces__msg__Metadata), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(dros_common_interfaces__msg__Metadata));
  bool success = dros_common_interfaces__msg__Metadata__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
dros_common_interfaces__msg__Metadata__destroy(dros_common_interfaces__msg__Metadata * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    dros_common_interfaces__msg__Metadata__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
dros_common_interfaces__msg__Metadata__Sequence__init(dros_common_interfaces__msg__Metadata__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dros_common_interfaces__msg__Metadata * data = NULL;

  if (size) {
    data = (dros_common_interfaces__msg__Metadata *)allocator.zero_allocate(size, sizeof(dros_common_interfaces__msg__Metadata), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = dros_common_interfaces__msg__Metadata__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        dros_common_interfaces__msg__Metadata__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
dros_common_interfaces__msg__Metadata__Sequence__fini(dros_common_interfaces__msg__Metadata__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      dros_common_interfaces__msg__Metadata__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

dros_common_interfaces__msg__Metadata__Sequence *
dros_common_interfaces__msg__Metadata__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  dros_common_interfaces__msg__Metadata__Sequence * array = (dros_common_interfaces__msg__Metadata__Sequence *)allocator.allocate(sizeof(dros_common_interfaces__msg__Metadata__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = dros_common_interfaces__msg__Metadata__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
dros_common_interfaces__msg__Metadata__Sequence__destroy(dros_common_interfaces__msg__Metadata__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    dros_common_interfaces__msg__Metadata__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
dros_common_interfaces__msg__Metadata__Sequence__are_equal(const dros_common_interfaces__msg__Metadata__Sequence * lhs, const dros_common_interfaces__msg__Metadata__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!dros_common_interfaces__msg__Metadata__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
dros_common_interfaces__msg__Metadata__Sequence__copy(
  const dros_common_interfaces__msg__Metadata__Sequence * input,
  dros_common_interfaces__msg__Metadata__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(dros_common_interfaces__msg__Metadata);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    dros_common_interfaces__msg__Metadata * data =
      (dros_common_interfaces__msg__Metadata *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!dros_common_interfaces__msg__Metadata__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          dros_common_interfaces__msg__Metadata__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!dros_common_interfaces__msg__Metadata__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
