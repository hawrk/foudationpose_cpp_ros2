#!/bin/bash

IMAGE_BASE_NAME="easy_deploy_base_dev"
BUILT_IMAGE_TAG="nvidia_gpu_trt10_u2204_ros2"
EXTERNAL_TAG="--runtime nvidia"  # 可选，根据是否需要 GPU

script_dir="$( cd "$(dirname "$0")" && pwd )"
parent_dir="$( cd "$script_dir/../../.." && pwd )"
parent_dir_name="$(basename "$parent_dir")"

CONTAINER_NAME="ros2_easy_deploy_${parent_dir_name}"

is_container_exist() {
  local name="$1"
  if docker ps -a --filter "name=^/${name}$" --format "{{.Names}}" | grep -wq "$name"; then
    return 0
  else
    return 1
  fi
}

attach_container() {
  if is_container_exist ${CONTAINER_NAME}; then
    docker exec -it ${CONTAINER_NAME} bash
  else
    echo "Container ${CONTAINER_NAME} does not exist."
  fi
}

ros2_env() {

  local image_full_name="${IMAGE_BASE_NAME}:${BUILT_IMAGE_TAG}"

  if is_container_exist ${CONTAINER_NAME}; then
    echo "Container ${CONTAINER_NAME} already exists. Skipping creation."
    return 0
  fi

  docker run -itd --privileged \
             --device /dev/dri \
             --group-add video \
             -v /tmp/.X11-unix:/tmp/.X11-unix \
             --network bridge \
             --ipc host \
             -v ${parent_dir}:/workspace \
             -w /workspace \
             -v /dev/bus/usb:/dev/bus/usb \
             -e DISPLAY=${DISPLAY} \
             -e DOCKER_USER=${USER} \
             -e USER=${USER} \
             -e ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-1} \
             -e ROS_LOCALHOST_ONLY=0 \
             --name ${CONTAINER_NAME} \
             ${EXTERNAL_TAG} \
             ${image_full_name} \
             /bin/bash

  echo "Container created. Attaching..."
  docker exec -it ${CONTAINER_NAME} bash
  return 0
}

if [ $# -gt 0 ]; then
  # 执行指定的函数
  "$@"
else
  ros2_env
fi
