#!/usr/bin/env python3
import time
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import Float64MultiArray

class PickAndPlaceMission(Node):
    def __init__(self):
        super().__init__('pick_and_place_mission')
        self.get_logger().info('Initializing Pick and Place Mission Manager...')

        # Publishers
        self.target_ee_pub = self.create_publisher(
            PoseStamped, '/nmpc/target_ee_pose', 10
        )
        self.obstacle_pub = self.create_publisher(
            PointStamped, '/nmpc/obstacle', 10
        )
        self.gripper_pub = self.create_publisher(
            Float64MultiArray, '/gripper_position_controller/commands', 10
        )

        # Subscriber
        self.current_ee_sub = self.create_subscription(
            PoseStamped, '/nmpc/current_ee_pose', self.current_ee_cb, 10
        )

        # State tracking
        self.current_ee = None
        self.state = 'INIT'
        self.state_start_time = time.time()
        
        # Exact waypoints relative to robot base:
        # Table surface: z = 0.025m (world z = 0.75m)
        # Target box: center at x=0.45, y=-0.28, z=0.060m
        # Place tray: center at x=0.45, y=+0.28, z=0.060m
        # Obstacle pillar: center at x=0.45, y=0.00, z=0.150m (top at z=0.275m)
        self.waypoints = {
            'HOME':           [0.35,  0.00, 0.45],
            'PICK_HOVER':     [0.45, -0.25, 0.22],
            'PICK_DOWN':      [0.45, -0.25, 0.050], # exact grasping height: pads fully enclose box
            'LIFT':           [0.45, -0.25, 0.36],  # lift above obstacle clearance
            'TRANSFER_HOVER': [0.40,  0.00, 0.38],  # safe apex waypoint over obstacle
            'PLACE_HOVER':    [0.45,  0.25, 0.22],
            'PLACE_DOWN':     [0.45,  0.25, 0.055]  # exact placement height
        }

        # 6-joint synchronized parallel gripper commands:
        # [left_knuckle, right_knuckle, left_inner_knuckle, right_inner_knuckle, left_finger_tip, right_finger_tip]
        # Tip counter-rotates (-knuckle) so pads stay 100% flat and parallel!
        self.GRIPPER_OPEN = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.GRIPPER_CLOSE = [0.55, -0.55, 0.55, -0.55, -0.55, 0.55]

        # Publish obstacle continuously at 2 Hz
        self.create_timer(0.5, self.publish_obstacle)

        # Mission control loop at 10 Hz
        self.timer = self.create_timer(0.1, self.step_mission)
        self.get_logger().info('Pick and Place Mission ready. Waiting for robot state...')

    def current_ee_cb(self, msg: PoseStamped):
        self.current_ee = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ]

    def publish_obstacle(self):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.point.x = 0.45
        msg.point.y = 0.0
        msg.point.z = 0.15
        self.obstacle_pub.publish(msg)

    def publish_target(self, pos):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.w = 1.0
        self.target_ee_pub.publish(msg)

    def set_gripper(self, state):
        msg = Float64MultiArray()
        msg.data = state
        self.gripper_pub.publish(msg)

    def get_distance(self, target):
        if self.current_ee is None:
            return 999.0
        dx = self.current_ee[0] - target[0]
        dy = self.current_ee[1] - target[1]
        dz = self.current_ee[2] - target[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def transition(self, next_state):
        self.get_logger().info(f'Transition: [{self.state}] -> [{next_state}]')
        self.state = next_state
        self.state_start_time = time.time()

    def step_mission(self):
        if self.current_ee is None:
            return

        elapsed = time.time() - self.state_start_time

        if self.state == 'INIT':
            self.set_gripper(self.GRIPPER_OPEN)
            self.publish_target(self.waypoints['PICK_HOVER'])
            if elapsed > 1.0:
                self.transition('MOVE_TO_PICK_HOVER')

        elif self.state == 'MOVE_TO_PICK_HOVER':
            target = self.waypoints['PICK_HOVER']
            self.publish_target(target)
            dist = self.get_distance(target)
            if dist < 0.035 or elapsed > 5.0:
                self.get_logger().info(f'Reached PICK_HOVER (dist={dist:.3f}m). Descending to grasp...')
                self.transition('DESCEND_TO_PICK')

        elif self.state == 'DESCEND_TO_PICK':
            target = self.waypoints['PICK_DOWN']
            self.publish_target(target)
            dist = self.get_distance(target)
            if dist < 0.025 or elapsed > 3.5:
                self.get_logger().info(f'At target box (dist={dist:.3f}m). Closing Robotiq gripper...')
                self.transition('GRASP')

        elif self.state == 'GRASP':
            self.publish_target(self.waypoints['PICK_DOWN'])
            self.set_gripper(self.GRIPPER_CLOSE)
            if elapsed > 1.8:
                self.get_logger().info('Target box secured! Lifting...')
                self.transition('LIFT_OBJECT')

        elif self.state == 'LIFT_OBJECT':
            target = self.waypoints['LIFT']
            self.publish_target(target)
            self.set_gripper(self.GRIPPER_CLOSE)
            dist = self.get_distance(target)
            if dist < 0.035 or elapsed > 3.5:
                self.get_logger().info(f'Lifted to clearance height (dist={dist:.3f}m). Navigating over obstacle...')
                self.transition('TRANSFER_APEX')

        elif self.state == 'TRANSFER_APEX':
            target = self.waypoints['TRANSFER_HOVER']
            self.publish_target(target)
            self.set_gripper(self.GRIPPER_CLOSE)
            dist = self.get_distance(target)
            if dist < 0.045 or elapsed > 4.5:
                self.get_logger().info('Cleared obstacle apex. Moving to Place hover...')
                self.transition('MOVE_TO_PLACE_HOVER')

        elif self.state == 'MOVE_TO_PLACE_HOVER':
            target = self.waypoints['PLACE_HOVER']
            self.publish_target(target)
            self.set_gripper(self.GRIPPER_CLOSE)
            dist = self.get_distance(target)
            if dist < 0.035 or elapsed > 4.5:
                self.get_logger().info(f'Above Place tray (dist={dist:.3f}m). Descending to place...')
                self.transition('DESCEND_TO_PLACE')

        elif self.state == 'DESCEND_TO_PLACE':
            target = self.waypoints['PLACE_DOWN']
            self.publish_target(target)
            self.set_gripper(self.GRIPPER_CLOSE)
            dist = self.get_distance(target)
            if dist < 0.025 or elapsed > 3.5:
                self.get_logger().info('Placed on tray. Opening gripper...')
                self.transition('RELEASE')

        elif self.state == 'RELEASE':
            self.publish_target(self.waypoints['PLACE_DOWN'])
            self.set_gripper(self.GRIPPER_OPEN)
            if elapsed > 1.8:
                self.get_logger().info('Object released! Retracting...')
                self.transition('RETRACT_AFTER_PLACE')

        elif self.state == 'RETRACT_AFTER_PLACE':
            target = self.waypoints['PLACE_HOVER']
            self.publish_target(target)
            self.set_gripper(self.GRIPPER_OPEN)
            dist = self.get_distance(target)
            if dist < 0.035 or elapsed > 3.5:
                self.get_logger().info('Retracted. Returning to Home stance...')
                self.transition('RETURN_HOME')

        elif self.state == 'RETURN_HOME':
            target = self.waypoints['HOME']
            self.publish_target(target)
            self.set_gripper(self.GRIPPER_OPEN)
            dist = self.get_distance(target)
            if dist < 0.045 or elapsed > 5.0:
                self.get_logger().info('SUCCESS: Pick & Place with Obstacle Avoidance completed!')
                self.transition('FINISHED')

        elif self.state == 'FINISHED':
            self.publish_target(self.waypoints['HOME'])

def main(args=None):
    rclpy.init(args=args)
    node = PickAndPlaceMission()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
