/*
 * @Author: hawrkchen
 * @Date: 2025-01-06 16:08:13
 * @LastEditTime: 2025-01-06 16:09:43
 * @Description: 
 * @FilePath: /data_collection_service/include/realsense/inc/ros_utils.hpp
 */

#pragma once


#include <iostream>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "librealsense2/rs.hpp"
#include "librealsense2/rsutil.h"


namespace realsense2_camera
{
    typedef std::pair<rs2_stream, int> stream_index_pair;

    const stream_index_pair COLOR{RS2_STREAM_COLOR, 0};
    const stream_index_pair DEPTH{RS2_STREAM_DEPTH, 0};
    const stream_index_pair INFRA0{RS2_STREAM_INFRARED, 0};
    const stream_index_pair INFRA1{RS2_STREAM_INFRARED, 1};
    const stream_index_pair INFRA2{RS2_STREAM_INFRARED, 2};
    const stream_index_pair GYRO{RS2_STREAM_GYRO, 0};
    const stream_index_pair ACCEL{RS2_STREAM_ACCEL, 0};

    bool isValidCharInName(char c);

    std::string rs2_to_ros(std::string rs2_name);
    std::string ros_stream_to_string(rs2_stream stream);
    std::string create_graph_resource_name(const std::string &original_name);
    const rmw_qos_profile_t qos_string_to_qos(std::string str);
    const std::string list_available_qos_strings();
    rs2_format string_to_rs2_format(std::string str);

}
