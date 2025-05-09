/*
 * @Author: hawrkchen
 * @Date: 2025-01-06 14:32:41
 * @LastEditTime: 2025-01-06 15:45:56
 * @FilePath: /data_collection_service/src/main.cpp
 * @Description:
 */

#include <memory>

#include "rclcpp/rclcpp.hpp"

#include "data_collection_service/data_collection_manager.hpp"


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto manager = std::make_shared<DataCollectionManager>(DCM_NODE_NAME);

    std::thread spin_thread([&]() {
        if(!manager->Init()) {
            RCLCPP_ERROR(rclcpp::get_logger(DCM_NODE_NAME), "Data Collection Manager initialized failed");
        }
    });
    spin_thread.detach();

    manager->Run();

    return 0;
}