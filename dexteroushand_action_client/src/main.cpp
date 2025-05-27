/*
 * @Author: hawrkchen
 * @Date: 2025-05-27 14:23:44
 * @LastEditors: Do not edit
 * @LastEditTime: 2025-05-27 14:41:02
 * @Description: 
 * @FilePath: /dexteroushand_action_client/src/main.cpp
 */
#include <rclcpp/rclcpp.hpp>

#include "rclcpp_action/rclcpp_action.hpp"

#include "dros_common_interfaces/action/dexterous_hand.hpp"

using DexterousHand = dros_common_interfaces::action::DexterousHand;

class DexterousHandClient : public rclcpp::Node {
  public: 
    DexterousHandClient(const std::string& node_name) :Node(node_name) {
        RCLCPP_INFO(this->get_logger(), "Creating DexterousHandClient start...");
        this->action_client_ = rclcpp_action::create_client<DexterousHand>(this, "hand_control_module");

        
        while(!this->action_client_ ->wait_for_action_server(std::chrono::seconds(1))) {
            if (!rclcpp::ok()) {
                RCLCPP_ERROR(this->get_logger(), "Action server not available after waiting");
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Waiting for action server to be up...");
        }
        RCLCPP_INFO(this->get_logger(), "Action server is up, creating goal request...");
    }

    void action_send() {
        auto goal_msg = DexterousHand::Goal();

        // fill in goal message here
        goal_msg.target_position = 5;
        goal_msg.obj_name = "water bottle";

        auto send_goal_options = rclcpp_action::Client<DexterousHand>::SendGoalOptions();
        send_goal_options.goal_response_callback =
            std::bind(&DexterousHandClient::goal_response_callback, this, std::placeholders::_1);
        send_goal_options.feedback_callback =
            std::bind(&DexterousHandClient::feedback_callback, this, std::placeholders::_1, std::placeholders::_2);     
        send_goal_options.result_callback =
            std::bind(&DexterousHandClient::result_callback, this, std::placeholders::_1);

        this->action_client_->async_send_goal(goal_msg, send_goal_options);

    }

  private:

    // action 请求回调
    void goal_response_callback(const rclcpp_action::ClientGoalHandle<DexterousHand>::SharedPtr& goal_handle) {
        if (!goal_handle) {
            RCLCPP_ERROR(this->get_logger(), "Goal rejected");
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Goal accepted");
    }

    // action 反馈回调
    void feedback_callback(const rclcpp_action::ClientGoalHandle<DexterousHand>::SharedPtr, 
                const std::shared_ptr<const DexterousHand::Feedback>& feedback) {
        RCLCPP_INFO(this->get_logger(), "Received feedback");
        float current_pose = feedback->progress;
        RCLCPP_INFO(this->get_logger(), "Current pose: %f", current_pose);
    }

    // action 结果回调
    void result_callback(const rclcpp_action::ClientGoalHandle<DexterousHand>::WrappedResult& result) {
        RCLCPP_INFO(this->get_logger(), "Received result");
        switch(result.code) {
            case rclcpp_action::ResultCode::SUCCEEDED:
                RCLCPP_INFO(this->get_logger(), "Action succeeded");
                break;
            case rclcpp_action::ResultCode::ABORTED:
                RCLCPP_INFO(this->get_logger(), "Action aborted");
                return;
            case rclcpp_action::ResultCode::CANCELED:
                RCLCPP_INFO(this->get_logger(), "Action canceled");
                return;
            default:
                RCLCPP_INFO(this->get_logger(), "Action returned with unknown result code");
                return;
        }
        RCLCPP_INFO(this->get_logger(), "Result: %d, msg:%s", 
            result.result->success, result.result->msg.c_str());
    }

  private:
    rclcpp_action::Client<DexterousHand>::SharedPtr action_client_;

};



int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DexterousHandClient>("dexterous_hand_client");
    node->action_send();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
