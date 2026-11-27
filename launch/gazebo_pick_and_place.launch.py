import os
import tempfile
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _launch_setup(context, *args, **kwargs):
    pkg_dsr_m0609_nmpc = get_package_share_directory('dsr_m0609_nmpc')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world_name = context.launch_configurations.get('world_name', 'pick_and_place_rail.sdf')
    world_file = os.path.join(pkg_dsr_m0609_nmpc, 'worlds', world_name)
    urdf_file = os.path.join(pkg_dsr_m0609_nmpc, 'model', 'm0609_gz_robotiq.urdf')

    with open(urdf_file, 'r') as f:
        robot_desc = f.read()

    # Resolve package placeholder in URDF dynamically for any user/workspace location
    robot_desc = robot_desc.replace('$(find dsr_m0609_nmpc)', pkg_dsr_m0609_nmpc)
    resolved_urdf_file = os.path.join(tempfile.gettempdir(), f'm0609_gz_robotiq_{os.getuid()}.urdf')
    with open(resolved_urdf_file, 'w') as f:
        f.write(robot_desc)

    # 1. Start Gazebo Sim with chosen world (static or dynamic obstacle)
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r -v 3 {world_file}'}.items()
    )

    # 2. Spawn Robot on the workbench table (z=0.750m is top surface of table)
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'm0609',
            '-file', resolved_urdf_file,
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.750'
        ],
        output='screen'
    )

    # 3. Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_desc}],
        output='screen'
    )

    # 4. ROS-Gazebo Bridge (clock, joint_states, D435 RGB-D topics, Conveyor TrackController)
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            '/d435/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/d435/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/d435/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/d435/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
            '/model/conveyor_feeder/link/conveyor_link/track_cmd_vel@std_msgs/msg/Float64]gz.msgs.Double',
        ],
        remappings=[
            ('/d435/image', '/camera/color/image_raw'),
            ('/d435/depth_image', '/camera/aligned_depth_to_color/image_raw'),
            ('/d435/camera_info', '/camera/color/camera_info'),
            ('/d435/points', '/camera/depth/color/points'),
            ('/model/conveyor_feeder/link/conveyor_link/track_cmd_vel', '/conveyor/cmd_vel'),
        ],
        output='screen'
    )

    # 4b. Static TF2 transforms for D435 Camera Frame (matches SDF pose 0.82 -0.05 2.04 0 1.24 3.14159265)
    tf_base_to_d435 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '0.82', '--y', '-0.05', '--z', '1.29',
            '--roll', '0.0', '--pitch', '1.24', '--yaw', '3.14159265',
            '--frame-id', 'base_link', '--child-frame-id', 'd435_link'
        ],
        output='screen'
    )

    tf_d435_to_optical = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '0.0', '--y', '0.0', '--z', '0.0',
            '--roll', '-1.5707963', '--pitch', '0.0', '--yaw', '-1.5707963',
            '--frame-id', 'd435_link', '--child-frame-id', 'd435_color_optical_frame'
        ],
        output='screen'
    )

    # 5. Controller Spawners
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen"
    )

    dsr_position_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["dsr_position_controller", "--controller-manager", "/controller_manager"],
        output="screen"
    )

    gripper_position_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_position_controller", "--controller-manager", "/controller_manager"],
        output="screen"
    )

    delay_controllers = TimerAction(
        period=4.0,
        actions=[
            joint_state_broadcaster_spawner,
            dsr_position_controller_spawner,
            gripper_position_controller_spawner
        ]
    )

    return [
        gz_sim,
        spawn_robot,
        robot_state_publisher,
        bridge,
        tf_base_to_d435,
        tf_d435_to_optical,
        delay_controllers,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'world_name',
            default_value='pick_and_place_rail.sdf',
            description='SDF world file in worlds/ (pick_and_place_rail.sdf or pick_and_place_dynamic.sdf)',
        ),
        OpaqueFunction(function=_launch_setup),
    ])
