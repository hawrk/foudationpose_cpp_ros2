/*
 * @Author: hawrkchen
 * @Date: 2025-05-15 15:16:29
 * @LastEditors: Do not edit
 * @LastEditTime: 2025-05-15 15:28:35
 * @Description: 
 * @FilePath: /grasp_client_ws/src/main.cpp
 */
#include <rclcpp/rclcpp.hpp>
#include <dros_common_interfaces/srv/grasp.hpp>

using Grasp = dros_common_interfaces::srv::Grasp;

class GraspClientNode : public rclcpp::Node {
  public:
    GraspClientNode(const std::string& node_name): Node(node_name) {
      RCLCPP_INFO(this->get_logger(), "Creating grasp client node");

      this->client_ = this->create_client<Grasp>("grasp");
      while(!this->client_->wait_for_service(std::chrono::seconds(1))) {
        RCLCPP_INFO(this->get_logger(), "Waiting for grasp service to be available...");
      }

      auto request = std::make_shared<Grasp::Request>();
      request->object_name = "water bottle";

      auto result_future = this->client_->async_send_request(request);
      RCLCPP_INFO(this->get_logger(), "我在等待抓取结果..");
      if(rclcpp::spin_until_future_complete(this->get_node_base_interface(), result_future) == rclcpp::FutureReturnCode::SUCCESS) {
        auto result = result_future.get();
        RCLCPP_INFO(this->get_logger(), "抓取结果: %d:%s", result->err_code, result->err_msg.c_str());
      }

    }


  private:
    rclcpp::Client<Grasp>::SharedPtr client_;

};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<GraspClientNode>("grasp_client");
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
