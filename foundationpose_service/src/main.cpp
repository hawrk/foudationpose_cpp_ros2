#include <rclcpp/rclcpp.hpp>
#include "dros_common_interfaces/msg/rgbd.hpp"


#include "detection_6d_foundationpose/foundationpose.hpp"
#include "trt_core/trt_core.h"

using namespace inference_core;
using namespace detection_6d;

static const std::string refiner_engine_path_ = "../models/refiner_hwc_dynamic_fp16.engine";
static const std::string scorer_engine_path_  = "../models/scorer_hwc_dynamic_fp16.engine";
static const std::string demo_data_path_      = "../test_data/mustard0";
static const std::string demo_textured_obj_path = demo_data_path_ + "/mesh/textured_simple.obj";
// static const std::string demo_textured_map_path = demo_data_path_ + "/mesh/texture_map.png";
static const std::string demo_name_             = "mustard";
static const std::string frame_id               = "1581120424100262102";
static const size_t      refine_itr             = 1;

class FoundationPoseService : public rclcpp::Node
{
  public:
    FoundationPoseService(const std::string& name):Node(name) {
      RCLCPP_INFO(this->get_logger(), "启动节点 :%s", name.c_str());

      sub_rgbd_ = this->create_subscription<dros_common_interfaces::msg::RGBD>(
        "camera/rgbd", 10, std::bind(&FoundationPoseService::rgbd_callback, this, std::placeholders::_1));

      // 初始化参数
      this->declare_parameter<std::string>("refiner_engine_path", "../models/refiner_hwc_dynamic_fp16.engine");
      this->declare_parameter<std::string>("scorer_engine_path", "../models/scorer_hwc_dynamic_fp16.engine");
      this->declare_parameter<std::string>("object_name", "mustard");
      this->declare_parameter<std::string>("object_mesh_path", "../test_data/mustard0/mesh/textured_simple.obj");
      this->declare_parameter<int>("refine_iterations", 1);
      this->declare_parameter<std::string>("camera_intrinsics_path", "../test_data/mustard0/cam_K.txt");

      refiner_engine_path_ = this->get_parameter("refiner_engine_path").as_string();
      scorer_engine_path_ = this->get_parameter("scorer_engine_path").as_string();
      demo_name_ = this->get_parameter("object_name").as_string();
      demo_textured_obj_path_ = this->get_parameter("object_mesh_path").as_string();
      refine_itr_ = this->get_parameter("refine_iterations").as_int();
      camera_intrinsics_path_ = this->get_parameter("camera_intrinsics_path").as_string();

      // 发布检测结果
      pub_pose_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("foundationpose/pose", 10);
      pub_debug_image_ = this->create_publisher<sensor_msgs::msg::Image>("foundationpose/debug_image", 10);
      
    }

  private:
    rclcpp::Subscription<dros_common_interfaces::msg::RGBD>::SharedPtr sub_rgbd_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_pose_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_debug_image_;

    // FoundationPose相关
    std::shared_ptr<Base6DofDetectionModel> foundation_pose_;
    std::shared_ptr<BaseMeshLoader> mesh_loader_;
    Eigen::Matrix3f intrinsic_in_mat_;
    Eigen::Vector3f object_dimension_;
    Eigen::Matrix4f last_pose_;
    bool is_initialized_ = false;

    // 参数
    std::string refiner_engine_path_;
    std::string scorer_engine_path_;
    std::string demo_name_;
    std::string demo_textured_obj_path_;
    std::string camera_intrinsics_path_;
    int refine_itr_;

    void rgbd_callback(const dros_common_interfaces::msg::RGBD::SharedPtr msg) {    
      RCLCPP_INFO(this->get_logger(), "收到RGBD数据");
      // TODO: 处理RGBD数据
      RCLCPP_INFO(this->get_logger(), "get frame_id:%s", msg->header.frame_id.c_str());
      RCLCPP_INFO(this->get_logger(), "get stamp:%d", msg->header.stamp.sec);
    }

};


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<FoundationPoseService>("foundationpose_service");
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
