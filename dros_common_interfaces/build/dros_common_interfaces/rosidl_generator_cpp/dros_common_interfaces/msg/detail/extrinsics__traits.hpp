// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from dros_common_interfaces:msg/Extrinsics.idl
// generated code does not contain a copyright notice

#ifndef DROS_COMMON_INTERFACES__MSG__DETAIL__EXTRINSICS__TRAITS_HPP_
#define DROS_COMMON_INTERFACES__MSG__DETAIL__EXTRINSICS__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "dros_common_interfaces/msg/detail/extrinsics__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace dros_common_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const Extrinsics & msg,
  std::ostream & out)
{
  out << "{";
  // member: rotation
  {
    if (msg.rotation.size() == 0) {
      out << "rotation: []";
    } else {
      out << "rotation: [";
      size_t pending_items = msg.rotation.size();
      for (auto item : msg.rotation) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: translation
  {
    if (msg.translation.size() == 0) {
      out << "translation: []";
    } else {
      out << "translation: [";
      size_t pending_items = msg.translation.size();
      for (auto item : msg.translation) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Extrinsics & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: rotation
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.rotation.size() == 0) {
      out << "rotation: []\n";
    } else {
      out << "rotation:\n";
      for (auto item : msg.rotation) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: translation
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.translation.size() == 0) {
      out << "translation: []\n";
    } else {
      out << "translation:\n";
      for (auto item : msg.translation) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Extrinsics & msg, bool use_flow_style = false)
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
  const dros_common_interfaces::msg::Extrinsics & msg,
  std::ostream & out, size_t indentation = 0)
{
  dros_common_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use dros_common_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const dros_common_interfaces::msg::Extrinsics & msg)
{
  return dros_common_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<dros_common_interfaces::msg::Extrinsics>()
{
  return "dros_common_interfaces::msg::Extrinsics";
}

template<>
inline const char * name<dros_common_interfaces::msg::Extrinsics>()
{
  return "dros_common_interfaces/msg/Extrinsics";
}

template<>
struct has_fixed_size<dros_common_interfaces::msg::Extrinsics>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<dros_common_interfaces::msg::Extrinsics>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<dros_common_interfaces::msg::Extrinsics>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // DROS_COMMON_INTERFACES__MSG__DETAIL__EXTRINSICS__TRAITS_HPP_
