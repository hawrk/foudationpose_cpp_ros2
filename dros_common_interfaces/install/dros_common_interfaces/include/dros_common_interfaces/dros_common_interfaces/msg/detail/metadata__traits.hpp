// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from dros_common_interfaces:msg/Metadata.idl
// generated code does not contain a copyright notice

#ifndef DROS_COMMON_INTERFACES__MSG__DETAIL__METADATA__TRAITS_HPP_
#define DROS_COMMON_INTERFACES__MSG__DETAIL__METADATA__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "dros_common_interfaces/msg/detail/metadata__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace dros_common_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const Metadata & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: json_data
  {
    out << "json_data: ";
    rosidl_generator_traits::value_to_yaml(msg.json_data, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Metadata & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: json_data
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "json_data: ";
    rosidl_generator_traits::value_to_yaml(msg.json_data, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Metadata & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace dros_common_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use dros_common_interfaces::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const dros_common_interfaces::msg::Metadata & msg,
  std::ostream & out, size_t indentation = 0)
{
  dros_common_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dros_common_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const dros_common_interfaces::msg::Metadata & msg)
{
  return dros_common_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<dros_common_interfaces::msg::Metadata>()
{
  return "dros_common_interfaces::msg::Metadata";
}

template<>
inline const char * name<dros_common_interfaces::msg::Metadata>()
{
  return "dros_common_interfaces/msg/Metadata";
}

template<>
struct has_fixed_size<dros_common_interfaces::msg::Metadata>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<dros_common_interfaces::msg::Metadata>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<dros_common_interfaces::msg::Metadata>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // DROS_COMMON_INTERFACES__MSG__DETAIL__METADATA__TRAITS_HPP_
