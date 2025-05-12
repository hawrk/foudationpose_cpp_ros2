#include <rclcpp/rclcpp.hpp>
#include "dros_common_interfaces/msg/rgbd.hpp"
#include "auto_sam_interfaces/srv/segment_rgbd.hpp"

#include "geometry_msgs/msg/pose_stamped.hpp"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp>
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <glog/logging.h>

#include "detection_6d_foundationpose/foundationpose.hpp"
#include "trt_core/trt_core.h"

#include <filesystem>
#include <algorithm>

using namespace inference_core;
using namespace detection_6d;

static const std::string demo_name_ = "fdp_ros2";

class FoundationPoseService : public rclcpp::Node
{
  public:
    FoundationPoseService(const std::string& name):Node(name) {
      RCLCPP_INFO(this->get_logger(), "启动节点 :%s", name.c_str());

      client_ = this->create_client<auto_sam_interfaces::srv::SegmentRGBD>("segment_rgbd");
      while (!client_->wait_for_service(std::chrono::seconds(1))) {
        RCLCPP_INFO(this->get_logger(), "Waiting for SegmentRGBD service...");
      }

      sub_rgbd_ = this->create_subscription<dros_common_interfaces::msg::RGBD>(
        "camera/rgbd", 10, std::bind(&FoundationPoseService::rgbd_callback, this, std::placeholders::_1));

      // 初始化参数
      this->declare_parameter<std::string>("refiner_engine_path", "./models/refiner_hwc_dynamic_fp16.engine");
      this->declare_parameter<std::string>("scorer_engine_path", "./models/scorer_hwc_dynamic_fp16.engine");
      this->declare_parameter<int>("refine_iterations", 1);

      refiner_engine_path_ = this->get_parameter("refiner_engine_path").as_string();
      scorer_engine_path_ = this->get_parameter("scorer_engine_path").as_string();
      refine_itr_ = this->get_parameter("refine_iterations").as_int();

      for (const auto& mask_type : obj_type_to_grasp) {
        std::string cad_dir = template_dir + "/" + mask_type + "/cad";
        auto paths = mesh_file_path_valid(cad_dir);
        if (!paths.empty())
          mesh_files.push_back(paths.front());  // 只取第一个文件
      }

      // 发布检测结果
      pub_pose_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("foundationpose/pose", 10);
      pub_debug_image_ = this->create_publisher<sensor_msgs::msg::Image>("foundationpose/debug_image", 10);
      
    }

    std::vector<std::string> mesh_file_path_valid(const std::string& dir) {
      std::vector<std::string> paths;
      for (const auto& entry : std::filesystem::recursive_directory_iterator(dir)) {
        if (entry.is_regular_file()) {
          std::string ext = entry.path().extension().string();
          std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
          if (ext == ".obj" || ext == ".stl" || ext == ".ply") {
            paths.push_back(entry.path().string());
          }
        }
      }
      return paths;
    }

    std::tuple<std::shared_ptr<Base6DofDetectionModel>, std::shared_ptr<BaseMeshLoader>> CreateModel(
      const std::string& demo_textured_obj_path, const Eigen::Matrix3f intrinsic_in_mat)
    {
      auto refiner_core = CreateTrtInferCore(this->refiner_engine_path_,
                                            {
                                                {"transf_input", {252, 160, 160, 6}},
                                                {"render_input", {252, 160, 160, 6}},
                                            },
                                            {{"trans", {252, 3}}, {"rot", {252, 3}}}, 1);
      auto scorer_core  = CreateTrtInferCore(this->scorer_engine_path_,
                                            {
                                                {"transf_input", {252, 160, 160, 6}},
                                                {"render_input", {252, 160, 160, 6}},
                                            },
                                            {{"scores", {252, 1}}}, 1);
    
      auto mesh_loader = CreateAssimpMeshLoader(demo_name_, demo_textured_obj_path);
      CHECK(mesh_loader != nullptr);

      auto foundation_pose =
          CreateFoundationPoseModel(refiner_core, scorer_core, {mesh_loader}, intrinsic_in_mat);

      return {foundation_pose, mesh_loader};
    }

    
    void foundation_pose_main(
      const cv::Mat& rgb_img,
      const cv::Mat& depth_img,
      uint32_t stamp_sec,
      uint32_t stamp_nanosec)
    {
      
      if (!is_initialized_)
      {
        auto request = std::make_shared<auto_sam_interfaces::srv::SegmentRGBD::Request>();
        cv_bridge::CvImage color_msg(std_msgs::msg::Header(), "rgb8", rgb_img);
        cv_bridge::CvImage depth_msg(std_msgs::msg::Header(), "32FC1", depth_img);
    
        request->rgb = *color_msg.toImageMsg();
        request->depth = *depth_msg.toImageMsg();
        request->obj_type_to_track = {"cup_tripple", "EM2E_left", "water_ybs"};
    
        auto future_result = client_->async_send_request(
          request,
          std::bind(&FoundationPoseService::response_callback, this, std::placeholders::_1));

        is_initialized_ = true;
      }

      for (auto& [foundation_pose, mesh_loader] : this->model_to_track_) {
        foundation_pose->Track(rgb_img.clone(), depth_img, out_pose, demo_name_, track_pose);
        LOG(WARNING) << "Track Pose : " << track_pose;
      }

      // cv::Mat regist_plot = rgb.clone();
      // cv::cvtColor(regist_plot, regist_plot, cv::COLOR_RGB2BGR);
      // auto draw_pose = ConvertPoseMesh2BBox(out_pose, mesh_loader);
      // draw3DBoundingBox(intrinsic_in_mat, draw_pose, 480, 640, object_dimension, regist_plot);
      // cv::imshow("test_foundationpose_result", regist_plot);
      // cv::waitKey(20);

      // 发布 Pose
      geometry_msgs::msg::PoseStamped pose_msg;
      pose_msg.header.stamp.sec = stamp_sec;
      pose_msg.header.stamp.nanosec = stamp_nanosec;
      pose_msg.header.frame_id = "camera_color_optical_frame";  // 你也可以使用 msg->header.frame_id

      Eigen::Quaternionf q(track_pose.block<3, 3>(0, 0));
      Eigen::Vector3f t = track_pose.block<3, 1>(0, 3);
      pose_msg.pose.position.x = t(0);
      pose_msg.pose.position.y = t(1);
      pose_msg.pose.position.z = t(2);
      pose_msg.pose.orientation.x = q.x();
      pose_msg.pose.orientation.y = q.y();
      pose_msg.pose.orientation.z = q.z();
      pose_msg.pose.orientation.w = q.w();

      pub_pose_->publish(pose_msg);

      // // 发布调试图像
      // sensor_msgs::msg::Image::SharedPtr debug_img_msg =
      //   cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", rgb_img).toImageMsg();
      // debug_img_msg->header.stamp.sec = stamp_sec;
      // debug_img_msg->header.stamp.nanosec = stamp_nanosec;
      // debug_img_msg->header.frame_id = "camera_color_optical_frame";

      // pub_debug_image_->publish(debug_img_msg);
    }


  private:
    rclcpp::Client<auto_sam_interfaces::srv::SegmentRGBD>::SharedPtr client_;

    rclcpp::Subscription<dros_common_interfaces::msg::RGBD>::SharedPtr sub_rgbd_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_pose_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_debug_image_;

    std::vector<std::string> obj_type_to_grasp = {"cup_tripple", "EM2E_left", "water_ybs"};
    int type_num_track = 1;
    std::string template_dir = "../auto_sam_service/template/grasp_bottle";
    std::vector<std::string> mesh_files;

    // FoundationPose相关
    Eigen::Matrix4f last_pose_;
    bool is_initialized_ = false;
    std::vector<std::pair<std::shared_ptr<Base6DofDetectionModel>, std::shared_ptr<BaseMeshLoader>>> model_to_track_;

    // 参数
    std::string refiner_engine_path_;
    std::string scorer_engine_path_;
    int refine_itr_;

    Eigen::Matrix3f rgb_K;
    Eigen::Vector3f object_dimension;

    Eigen::Matrix4f out_pose;
    Eigen::Matrix4f track_pose;

    void rgbd_callback(const dros_common_interfaces::msg::RGBD::SharedPtr msg) {    
      RCLCPP_INFO(this->get_logger(), "收到RGBD数据");

      // TODO: 处理RGBD数据
      std_msgs::msg::Header header = msg->header;
      RCLCPP_INFO(this->get_logger(), "get frame_id:%s", header.frame_id.c_str());
      RCLCPP_INFO(this->get_logger(), "get stamp:%d", header.stamp.sec);
      
      const sensor_msgs::msg::CameraInfo& rgb_info = msg->rgb_camera_info;
      this->rgb_K << rgb_info.k[0], rgb_info.k[1], rgb_info.k[2],
               rgb_info.k[3], rgb_info.k[4], rgb_info.k[5],
               rgb_info.k[6], rgb_info.k[7], rgb_info.k[8];
      RCLCPP_INFO(this->get_logger(), "RGB Camera fx: %f, fy: %f, cx: %f, cy: %f",
                  rgb_info.k[0], rgb_info.k[4], rgb_info.k[2], rgb_info.k[5]);
      
      cv::Mat rgb_img;
      try {
        rgb_img = cv_bridge::toCvCopy(msg->rgb, sensor_msgs::image_encodings::RGB8)->image;
      } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge RGB 转换失败: %s", e.what());
        return;
      }
    
      cv::Mat depth_img;
      try {
        // depth_img = cv_bridge::toCvCopy(msg->depth, sensor_msgs::image_encodings::TYPE_16UC1)->image;
        // depth_img.convertTo(depth_img, CV_32FC1, 0.001);
        depth_img = cv_bridge::toCvCopy(msg->depth, sensor_msgs::image_encodings::TYPE_32FC1)->image;
      } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge Depth 转换失败: %s", e.what());
        return;
      }

      RCLCPP_INFO(this->get_logger(), "RGB 图尺寸: %dx%d", rgb_img.cols, rgb_img.rows);
      RCLCPP_INFO(this->get_logger(), "Depth 图尺寸: %dx%d", depth_img.cols, depth_img.rows);

      this->foundation_pose_main(rgb_img, depth_img, header.stamp.sec, header.stamp.nanosec);
    }

    void response_callback(rclcpp::Client<auto_sam_interfaces::srv::SegmentRGBD>::SharedFuture future)
    {
      auto response = future.get();
      RCLCPP_INFO(this->get_logger(), "Received %zu segmentation results.", response->results.size());
  
      for (size_t i = 0; i < response->results.size(); ++i) {
        const auto& result = response->results[i];
        cv::Mat mask = cv_bridge::toCvCopy(result.mask, "mono8")->image;
        cv::Mat color, depth;
        color = cv_bridge::toCvCopy(response->rgb_out, "rgb8")->image;
        depth = cv_bridge::toCvCopy(response->depth_out, "32FC1")->image;
  
        std::string filename = "./det_res/mask_" + std::to_string(i) + "_" + result.mask_type + ".png";
        cv::imwrite(filename, mask);
  
        RCLCPP_INFO(this->get_logger(),
          "[%zu] Label: %s | Mask Type: %s | Score: %.2f",
          i, result.label.c_str(), result.mask_type.c_str(), result.score);
        
        if (result.mask_type == this->obj_type_to_grasp[this->type_num_track]){
          auto [foundation_pose, mesh_loader] = this->CreateModel(mesh_files[this->type_num_track], this->rgb_K);
          foundation_pose->Register(color.clone(), depth, mask, demo_name_, out_pose, refine_itr_);
          LOG(WARNING) << "first Pose : " << out_pose;
          this->model_to_track_.emplace_back(foundation_pose, mesh_loader);
        }
      }
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
