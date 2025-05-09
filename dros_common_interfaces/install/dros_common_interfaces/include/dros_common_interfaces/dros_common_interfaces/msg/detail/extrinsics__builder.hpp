// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dros_common_interfaces:msg/Extrinsics.idl
// generated code does not contain a copyright notice

#ifndef DROS_COMMON_INTERFACES__MSG__DETAIL__EXTRINSICS__BUILDER_HPP_
#define DROS_COMMON_INTERFACES__MSG__DETAIL__EXTRINSICS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dros_common_interfaces/msg/detail/extrinsics__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dros_common_interfaces
{

namespace msg
{

namespace builder
{

class Init_Extrinsics_translation
{
public:
  explicit Init_Extrinsics_translation(::dros_common_interfaces::msg::Extrinsics & msg)
  : msg_(msg)
  {}
  ::dros_common_interfaces::msg::Extrinsics translation(::dros_common_interfaces::msg::Extrinsics::_translation_type arg)
  {
    msg_.translation = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dros_common_interfaces::msg::Extrinsics msg_;
};

class Init_Extrinsics_rotation
{
public:
  Init_Extrinsics_rotation()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Extrinsics_translation rotation(::dros_common_interfaces::msg::Extrinsics::_rotation_type arg)
  {
    msg_.rotation = std::move(arg);
    return Init_Extrinsics_translation(msg_);
  }

private:
  ::dros_common_interfaces::msg::Extrinsics msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::dros_common_interfaces::msg::Extrinsics>()
{
  return dros_common_interfaces::msg::builder::Init_Extrinsics_rotation();
}

}  // namespace dros_common_interfaces

#endif  // DROS_COMMON_INTERFACES__MSG__DETAIL__EXTRINSICS__BUILDER_HPP_
