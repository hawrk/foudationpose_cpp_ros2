// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dros_common_interfaces:msg/Metadata.idl
// generated code does not contain a copyright notice

#ifndef DROS_COMMON_INTERFACES__MSG__DETAIL__METADATA__BUILDER_HPP_
#define DROS_COMMON_INTERFACES__MSG__DETAIL__METADATA__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dros_common_interfaces/msg/detail/metadata__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dros_common_interfaces
{

namespace msg
{

namespace builder
{

class Init_Metadata_json_data
{
public:
  explicit Init_Metadata_json_data(::dros_common_interfaces::msg::Metadata & msg)
  : msg_(msg)
  {}
  ::dros_common_interfaces::msg::Metadata json_data(::dros_common_interfaces::msg::Metadata::_json_data_type arg)
  {
    msg_.json_data = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dros_common_interfaces::msg::Metadata msg_;
};

class Init_Metadata_header
{
public:
  Init_Metadata_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Metadata_json_data header(::dros_common_interfaces::msg::Metadata::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_Metadata_json_data(msg_);
  }

private:
  ::dros_common_interfaces::msg::Metadata msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::dros_common_interfaces::msg::Metadata>()
{
  return dros_common_interfaces::msg::builder::Init_Metadata_header();
}

}  // namespace dros_common_interfaces

#endif  // DROS_COMMON_INTERFACES__MSG__DETAIL__METADATA__BUILDER_HPP_
