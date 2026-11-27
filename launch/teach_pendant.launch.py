import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_dsr_m0609_nmpc = get_package_share_directory('dsr_m0609_nmpc')

    teach_pendant_gui_node = Node(
        package='dsr_m0609_nmpc',
        executable='dsr_teach_pendant_gui.py',
        name='dsr_teach_pendant_gui',
        output='screen'
    )

    return LaunchDescription([
        teach_pendant_gui_node
    ])
