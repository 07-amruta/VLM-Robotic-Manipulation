cmake_minimum_required(VERSION 3.8)
project(perception_msgs)
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/InstanceMask.msg" "msg/InstanceMaskArray.msg"
  "msg/SceneObject.msg" "msg/SceneObjectArray.msg"
  DEPENDENCIES std_msgs sensor_msgs)
ament_export_dependencies(rosidl_default_runtime)
ament_package()