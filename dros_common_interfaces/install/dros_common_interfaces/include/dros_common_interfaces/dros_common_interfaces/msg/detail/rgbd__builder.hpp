// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dros_common_interfaces:msg/RGBD.idl
// generated code does not contain a copyright notice

#ifndef DROS_COMMON_INTERFACES__MSG__DETAIL__RGBD__BUILDER_HPP_
#define DROS_COMMON_INTERFACES__MSG__DETAIL__RGBD__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dros_common_interfaces/msg/detail/rgbd__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dros_common_interfaces
{

namespace msg
{

namespace builder
{

class Init_RGBD_depth
{
public:
  explicit Init_RGBD_depth(::dros_common_interfaces::msg::RGBD & msg)
  : msg_(msg)
  {}
  ::dros_common_interfaces::msg::RGBD depth(::dros_common_interfaces::msg::RGBD::_depth_type arg)
  {
    msg_.depth = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dros_common_interfaces::msg::RGBD msg_;
};

class Init_RGBD_rgb
{
public:
  explicit Init_RGBD_rgb(::dros_common_interfaces::msg::RGBD & msg)
  : msg_(msg)
  {}
  Init_RGBD_depth rgb(::dros_common_interfaces::msg::RGBD::_rgb_type arg)
  {
    msg_.rgb = std::move(arg);
    return Init_RGBD_depth(msg_);
  }

private:
  ::dros_common_interfaces::msg::RGBD msg_;
};

class Init_RGBD_depth_camera_info
{
public:
  explicit Init_RGBD_depth_camera_info(::dros_common_interfaces::msg::RGBD & msg)
  : msg_(msg)
  {}
  Init_RGBD_rgb depth_camera_info(::dros_common_interfaces::msg::RGBD::_depth_camera_info_type arg)
  {
    msg_.depth_camera_info = std::move(arg);
    return Init_RGBD_rgb(msg_);
  }

private:
  ::dros_common_interfaces::msg::RGBD msg_;
};

class Init_RGBD_rgb_camera_info
{
public:
  explicit Init_RGBD_rgb_camera_info(::dros_common_interfaces::msg::RGBD & msg)
  : msg_(msg)
  {}
  Init_RGBD_depth_camera_info rgb_camera_info(::dros_common_interfaces::msg::RGBD::_rgb_camera_info_type arg)
  {
    msg_.rgb_camera_info = std::move(arg);
    return Init_RGBD_depth_camera_info(msg_);
  }

private:
  ::dros_common_interfaces::msg::RGBD msg_;
};

class Init_RGBD_header
{
public:
  Init_RGBD_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_RGBD_rgb_camera_info header(::dros_common_interfaces::msg::RGBD::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_RGBD_rgb_camera_info(msg_);
  }

private:
  ::dros_common_interfaces::msg::RGBD msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::dros_common_interfaces::msg::RGBD>()
{
  return dros_common_interfaces::msg::builder::Init_RGBD_header();
}

}  // namespace dros_common_interfaces

#endif  // DROS_COMMON_INTERFACES__MSG__DETAIL__RGBD__BUILDER_HPP_
