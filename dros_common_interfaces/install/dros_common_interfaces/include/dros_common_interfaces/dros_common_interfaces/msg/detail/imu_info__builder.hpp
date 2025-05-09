// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from dros_common_interfaces:msg/IMUInfo.idl
// generated code does not contain a copyright notice

#ifndef DROS_COMMON_INTERFACES__MSG__DETAIL__IMU_INFO__BUILDER_HPP_
#define DROS_COMMON_INTERFACES__MSG__DETAIL__IMU_INFO__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "dros_common_interfaces/msg/detail/imu_info__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace dros_common_interfaces
{

namespace msg
{

namespace builder
{

class Init_IMUInfo_bias_variances
{
public:
  explicit Init_IMUInfo_bias_variances(::dros_common_interfaces::msg::IMUInfo & msg)
  : msg_(msg)
  {}
  ::dros_common_interfaces::msg::IMUInfo bias_variances(::dros_common_interfaces::msg::IMUInfo::_bias_variances_type arg)
  {
    msg_.bias_variances = std::move(arg);
    return std::move(msg_);
  }

private:
  ::dros_common_interfaces::msg::IMUInfo msg_;
};

class Init_IMUInfo_noise_variances
{
public:
  explicit Init_IMUInfo_noise_variances(::dros_common_interfaces::msg::IMUInfo & msg)
  : msg_(msg)
  {}
  Init_IMUInfo_bias_variances noise_variances(::dros_common_interfaces::msg::IMUInfo::_noise_variances_type arg)
  {
    msg_.noise_variances = std::move(arg);
    return Init_IMUInfo_bias_variances(msg_);
  }

private:
  ::dros_common_interfaces::msg::IMUInfo msg_;
};

class Init_IMUInfo_data
{
public:
  explicit Init_IMUInfo_data(::dros_common_interfaces::msg::IMUInfo & msg)
  : msg_(msg)
  {}
  Init_IMUInfo_noise_variances data(::dros_common_interfaces::msg::IMUInfo::_data_type arg)
  {
    msg_.data = std::move(arg);
    return Init_IMUInfo_noise_variances(msg_);
  }

private:
  ::dros_common_interfaces::msg::IMUInfo msg_;
};

class Init_IMUInfo_header
{
public:
  Init_IMUInfo_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_IMUInfo_data header(::dros_common_interfaces::msg::IMUInfo::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_IMUInfo_data(msg_);
  }

private:
  ::dros_common_interfaces::msg::IMUInfo msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::dros_common_interfaces::msg::IMUInfo>()
{
  return dros_common_interfaces::msg::builder::Init_IMUInfo_header();
}

}  // namespace dros_common_interfaces

#endif  // DROS_COMMON_INTERFACES__MSG__DETAIL__IMU_INFO__BUILDER_HPP_
