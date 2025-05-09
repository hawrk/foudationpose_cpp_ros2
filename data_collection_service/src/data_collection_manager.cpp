/*
 * @Author: hawrkchen
 * @Date: 2025-01-06 14:32:36
 * @LastEditTime: 2025-01-06 15:46:08
 * @Description: 
 * @FilePath: /data_collection_service/src/data_collection_manager.cpp
 */


#include "data_collection_service/data_collection_manager.hpp"

DataCollectionManager::DataCollectionManager(const std::string& name) 
    :name_(name)
{
    RCLCPP_INFO(rclcpp::get_logger(DCM_NODE_NAME), "DataCollectionManager is constructed");
    node_ = rclcpp::Node::make_shared(name);
    executor_.add_node(node_);
}

DataCollectionManager::~DataCollectionManager() 
{  
    
}

bool DataCollectionManager::Init() 
{

    return true;
    
}

void DataCollectionManager::Run() 
{
    executor_.spin();
    rclcpp::shutdown();
}
