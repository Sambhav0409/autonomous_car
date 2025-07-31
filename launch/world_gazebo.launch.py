#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_share_dir = get_package_share_directory('self_driving_car_pkg')
    

    # Fix GAZEBO_MODEL_PATH
    models_share_dir = os.path.join(pkg_share_dir, 'models')
    if 'GAZEBO_MODEL_PATH' in os.environ:
        os.environ['GAZEBO_MODEL_PATH'] += os.pathsep + models_share_dir
    else:
        os.environ['GAZEBO_MODEL_PATH'] = models_share_dir

    print("[GAZEBO_MODEL_PATH] = ", os.environ['GAZEBO_MODEL_PATH'])

    world = os.path.join(pkg_share_dir, 'worlds', 'sdc.world')

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world}.items()
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )

    # Correct model paths
    red_light_sdf = os.path.join(pkg_share_dir, 'models', 'light_red', 'model.sdf')
    yellow_light_sdf = os.path.join(pkg_share_dir, 'models', 'light_yellow', 'model.sdf')
    green_light_sdf = os.path.join(pkg_share_dir, 'models', 'light_green', 'model.sdf')

    spawn_red_light = ExecuteProcess(
        cmd=['ros2', 'run', 'self_driving_car_pkg', 'spawner_node', red_light_sdf, 'red_light'],
        output='screen'
    )

    spawn_yellow_light = ExecuteProcess(
        cmd=['ros2', 'run', 'self_driving_car_pkg', 'spawner_node', yellow_light_sdf, 'yellow_light'],
        output='screen'
    )

    spawn_green_light = ExecuteProcess(
        cmd=['ros2', 'run', 'self_driving_car_pkg', 'spawner_node', green_light_sdf, 'green_light'],
        output='screen'
    )

    # Add ros_gz_bridge for camera topic bridging
    bridge_camera_image = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
            '/prius_hybrid/camera_frame/camera/image@sensor_msgs/msg/Image[gz.msgs.Image'
        ],
        output='screen'
    )

    ld = LaunchDescription()
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(spawn_red_light)
    ld.add_action(TimerAction(period=7.5, actions=[spawn_yellow_light]))
    ld.add_action(TimerAction(period=8.5, actions=[spawn_green_light]))
    ld.add_action(bridge_camera_image)

    return ld
