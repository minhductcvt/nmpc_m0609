from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='dsr_m0609_nmpc',
            executable='m0609_nmpc_node',
            name='m0609_nmpc_node',
            output='screen',
            parameters=[{
                'update_rate': 50.0,
            }]
        )
    ])
