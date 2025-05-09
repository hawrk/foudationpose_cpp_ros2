/*
 * @Author: hawrkchen
 * @Date: 2025-01-06 14:50:43
 * @LastEditTime: 2025-01-06 15:45:17
 * @Description: 
 * @FilePath: /data_collection_service/include/data_collection_service/data_collection_manager.hpp
 */


#pragma once

#include "rclcpp/rclcpp.hpp"

const std::string  DCM_NODE_NAME = "data_collection_manager";

class DataCollectionManager {
    public:
        DataCollectionManager(const std::string & name);
        ~DataCollectionManager();

        // 所有数据采集相关的sdk初始化工作
        bool Init();

        // ros node 运行
        void Run();

    private:
        std::string name_;
        rclcpp::Node::SharedPtr node_;
        rclcpp::executors::MultiThreadedExecutor executor_;
};