"""Launch the Smart Farm inspection node alongside a running sim stack.

    ros2 launch sim_bringup go2_sim.launch.py          # terminal 1
    ros2 launch smartfarm_inspector smartfarm.launch.py # terminal 2
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Generate the launch description."""
    args = [
        DeclareLaunchArgument(
            "auto_launch", default_value="true",
            description="Start the mission automatically once it validates. "
                        "Set false to review it in RViz first."),
        DeclareLaunchArgument(
            "backend", default_value="",
            description="agent_sdk (subscription) | api | offline. "
                        "Empty autodetects."),
        DeclareLaunchArgument("model", default_value="claude-sonnet-5"),
    ]

    node = Node(
        package="smartfarm_inspector",
        executable="inspection_node",
        name="smartfarm_inspection_node",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "auto_launch": LaunchConfiguration("auto_launch"),
            "backend": LaunchConfiguration("backend"),
            "model": LaunchConfiguration("model"),
            "project_id": "smartfarm",
            "default_battery_soc": 0.95,
        }],
    )
    return LaunchDescription(args + [node])
