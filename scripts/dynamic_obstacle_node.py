#!/usr/bin/env python3
"""
Dynamic Obstacle Generator & Gazebo Harmonic Pose Driver for Doosan M0609 NMPC.
Supports switching live between:
  - 'circle': Circular orbiting cylinder obstacle
  - 'zigzag': Unpredictable randomized back-and-forth zigzag corridor obstacle
  - 'none': Parks both dynamic obstacles off-table
"""

import math
import random
import sys
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool, Float64, Float64MultiArray, String

try:
    import gz.transport13 as gz_transport
    from gz.msgs10 import pose_pb2, boolean_pb2
    HAVE_GZ = True
except ImportError:
    HAVE_GZ = False


WORLD_NAME = "m0609_dynamic_obstacle_world"
SET_POSE_SVC = f"/world/{WORLD_NAME}/set_pose"
RAIL_SET_POSE_SVC = "/world/m0609_rail_world/set_pose"
TABLE_TOP_Z_WORLD = 0.750
OBS_HEIGHT = 0.260
RAIL_OBS_HEIGHT = 0.274
CIRCLE_RADIUS = 0.050
ZIGZAG_RADIUS = 0.055  # Effective radius for ~9x9cm yellow column
PARK_POSE = (2.5, 2.5, -1.0)


class DynamicObstacleNode(Node):
    def __init__(self, initial_mode: str = "circle"):
        super().__init__("dynamic_obstacle_node")
        self.declare_parameter("mode", initial_mode)
        self.declare_parameter("speed_scale", 1.0)

        self.mode = str(self.get_parameter("mode").value).strip().lower()
        if self.mode not in ("circle", "zigzag", "rail", "none"):
            self.mode = "circle"
        self.speed_scale = float(self.get_parameter("speed_scale").value)

        self.gz_node = gz_transport.Node() if HAVE_GZ else None
        if not HAVE_GZ:
            self.get_logger().warn("gz.transport13 not available — publishing ROS 2 obstacle topics only.")

        # ROS 2 Subscribers
        self.create_subscription(String, "/dynamic_obstacle/mode", self._mode_cb, 10)
        self.create_subscription(Float64, "/dynamic_obstacle/speed", self._speed_cb, 10)
        self.create_subscription(Bool, "/dynamic_obstacle/paused", self._paused_cb, 10)

        # ROS 2 Publishers
        self.state_pub = self.create_publisher(Float64MultiArray, "/dynamic_obstacle/state", 10)
        self.nmpc_obs_pub = self.create_publisher(PointStamped, "/nmpc/obstacle", 10)
        self.nmpc_obs_info_pub = self.create_publisher(Float64MultiArray, "/nmpc/obstacle_info", 10)
        self.vis_obs_info_pub = self.create_publisher(Float64MultiArray, "/vision/obstacle_info", 10)

        # Motion state
        self.paused = False
        self.t_last = time.monotonic()
        self.circle_theta = 0.0
        self.rail_phase = 0.0
        self.rng = random.Random()
        self.last_ox = 0.545
        self.last_oy = -0.140
        self.last_yaw = 0.0

        # Erratic Zigzag state strictly inside 30cm mid-zone [0.32..0.58, -0.24..-0.04] between HOME and conveyor
        self.zz_side = 1
        self.zz_pos = [0.45, -0.22]
        self.zz_start = [0.45, -0.22]
        self.zz_target = [0.52, -0.05]
        self.zz_seg_dur = 1.2
        self.zz_seg_t = 0.0
        self.zz_vel = [0.0, 0.0]
        self.zz_phase = 0.0
        self._pick_next_zigzag_segment()

        self._park_counter = 0
        self._needs_park_sync = True

        self.create_timer(1.0 / 30.0, self._step)
        self.get_logger().info(f"DynamicObstacleNode ready (mode={self.mode}, speed={self.speed_scale:.2f}x, gz={HAVE_GZ})")

    def _mode_cb(self, msg: String):
        req = msg.data.strip().lower()
        if req in ("circle", "zigzag", "rail", "none") and req != self.mode:
            self.mode = req
            self._needs_park_sync = True
            self.get_logger().info(f"Switched dynamic obstacle mode -> [{self.mode}]")

    def _speed_cb(self, msg: Float64):
        self.speed_scale = max(0.20, min(2.5, float(msg.data)))

    def _paused_cb(self, msg: Bool):
        new_p = bool(msg.data)
        if new_p != self.paused:
            self.paused = new_p
            self.t_last = time.monotonic()
            self.get_logger().info(f"Dynamic obstacle motion {'PAUSED' if self.paused else 'RESUMED'}")

    def _pick_next_zigzag_segment(self):
        self.zz_start = list(self.zz_pos)
        self.zz_side *= -1
        # Unpredictable bounce across Y inside the 30cm mid-zone [-0.24..-0.04]
        next_y = -0.14 + self.zz_side * self.rng.uniform(0.055, 0.095)
        if self.zz_pos[0] > 0.45:
            next_x = self.rng.uniform(0.32, 0.44)
        else:
            next_x = self.rng.uniform(0.46, 0.58)
        if self.rng.random() < 0.30:
            next_x = self.rng.uniform(0.33, 0.57)
        self.zz_target = [next_x, next_y]
        dist = math.hypot(next_x - self.zz_start[0], next_y - self.zz_start[1])
        base_speed = self.rng.uniform(0.16, 0.30) * self.speed_scale
        self.zz_seg_dur = max(0.45, dist / max(0.08, base_speed))
        self.zz_seg_t = 0.0
        self.zz_phase = self.rng.uniform(0.0, 2.0 * math.pi)

    def _send_gz_pose(self, model_name: str, x: float, y: float, z_world: float, yaw: float = 0.0, svc: str = SET_POSE_SVC):
        if not self.gz_node:
            return
        req = pose_pb2.Pose()
        req.name = model_name
        req.position.x = float(x)
        req.position.y = float(y)
        req.position.z = float(z_world)
        req.orientation.z = math.sin(yaw * 0.5)
        req.orientation.w = math.cos(yaw * 0.5)
        try:
            self.gz_node.request(svc, req, pose_pb2.Pose, boolean_pb2.Boolean, 40)
        except Exception:
            pass

    def _step(self):
        now = time.monotonic()
        dt = max(0.001, min(0.1, now - self.t_last))
        self.t_last = now
        self._park_counter += 1

        z_world_obs = TABLE_TOP_Z_WORLD + OBS_HEIGHT * 0.5  # 0.880 m
        obs_h_out = OBS_HEIGHT

        if self.mode == "rail":
            amp_x, freq = 0.075, 0.95 * self.speed_scale
            if not self.paused:
                self.rail_phase = (self.rail_phase + freq * dt) % (2.0 * math.pi)
                vx = amp_x * freq * math.cos(self.rail_phase)
            else:
                vx = 0.0
            ox = 0.50 + amp_x * math.sin(self.rail_phase)
            oy = 0.0
            vy = 0.0
            orad = CIRCLE_RADIUS
            obs_h_out = RAIL_OBS_HEIGHT
            mode_id = 3.0
            self.last_ox, self.last_oy, self.last_yaw = ox, oy, 0.0
            self._send_gz_pose("obstacle_rail_yellow", ox, oy, 0.894, 0.0, svc=RAIL_SET_POSE_SVC)

        elif self.mode == "circle":
            cx, cy, r_orb = 0.45, -0.14, 0.095
            omega = 1.25 * self.speed_scale
            if not self.paused:
                self.circle_theta = (self.circle_theta + omega * dt) % (2.0 * math.pi)
                vx = -r_orb * omega * math.sin(self.circle_theta)
                vy = r_orb * omega * math.cos(self.circle_theta)
            else:
                vx, vy = 0.0, 0.0
            ox = cx + r_orb * math.cos(self.circle_theta)
            oy = cy + r_orb * math.sin(self.circle_theta)
            orad = CIRCLE_RADIUS
            mode_id = 1.0
            self.last_ox, self.last_oy, self.last_yaw = ox, oy, self.circle_theta

            self._send_gz_pose("obstacle_circle", ox, oy, z_world_obs, self.circle_theta)
            if self._needs_park_sync or (self._park_counter % 45 == 0):
                self._send_gz_pose("obstacle_zigzag", *PARK_POSE, 0.0)
                self._needs_park_sync = False

        elif self.mode == "zigzag":
            if not self.paused:
                self.zz_seg_t += dt
                u = min(1.0, self.zz_seg_t / self.zz_seg_dur)
                prev_x, prev_y = self.zz_pos
                lx = self.zz_start[0] + u * (self.zz_target[0] - self.zz_start[0])
                ly = self.zz_start[1] + u * (self.zz_target[1] - self.zz_start[1])
                dx_seg = self.zz_target[0] - self.zz_start[0]
                dy_seg = self.zz_target[1] - self.zz_start[1]
                seg_len = max(1e-4, math.hypot(dx_seg, dy_seg))
                nx, ny = -dy_seg / seg_len, dx_seg / seg_len
                weave = 0.018 * math.sin(math.pi * u) * math.sin(2.0 * math.pi * u + self.zz_phase)

                ox = max(0.32, min(0.58, lx + weave * nx))
                oy = max(-0.24, min(-0.04, ly + weave * ny))
                vx = (ox - prev_x) / dt
                vy = (oy - prev_y) / dt
                self.zz_pos = [ox, oy]
                self.zz_vel = [vx, vy]
                yaw_vis = math.atan2(dy_seg, dx_seg)
                self.last_ox, self.last_oy, self.last_yaw = ox, oy, yaw_vis
                if u >= 1.0:
                    self._pick_next_zigzag_segment()
            else:
                ox, oy = self.zz_pos[0], self.zz_pos[1]
                vx, vy = 0.0, 0.0
                yaw_vis = self.last_yaw
            orad = ZIGZAG_RADIUS
            mode_id = 2.0

            self._send_gz_pose("obstacle_zigzag", ox, oy, z_world_obs, yaw_vis)
            if self._needs_park_sync or (self._park_counter % 45 == 0):
                self._send_gz_pose("obstacle_circle", *PARK_POSE, 0.0)
                self._needs_park_sync = False

        else:
            ox, oy, vx, vy, orad, mode_id = 2.0, 2.0, 0.0, 0.0, 0.04, 0.0
            if self._needs_park_sync or (self._park_counter % 45 == 0):
                self._send_gz_pose("obstacle_circle", *PARK_POSE, 0.0)
                self._send_gz_pose("obstacle_zigzag", *PARK_POSE, 0.0)
                self._needs_park_sync = False

        # Publish ROS 2 obstacle info + velocity for C++ NMPC and GUI Radar
        st = Float64MultiArray()
        st.data = [float(ox), float(oy), float(vx), float(vy), float(obs_h_out), float(orad), float(mode_id)]
        self.state_pub.publish(st)

        if self.mode != "none":
            pt = PointStamped()
            pt.header.stamp = self.get_clock().now().to_msg()
            pt.header.frame_id = "base_link"
            pt.point.x = float(ox)
            pt.point.y = float(oy)
            pt.point.z = float(obs_h_out * 0.5)
            self.nmpc_obs_pub.publish(pt)

            info_nmpc = Float64MultiArray()
            info_nmpc.data = [float(ox), float(oy), float(obs_h_out), float(orad), float(vx), float(vy)]
            self.nmpc_obs_info_pub.publish(info_nmpc)

            info_vis = Float64MultiArray()
            info_vis.data = [float(ox), float(oy), float(obs_h_out), float(orad)]
            self.vis_obs_info_pub.publish(info_vis)


def main(args=None):
    rclpy.init(args=args)
    init_mode = "circle"
    for i, a in enumerate(sys.argv):
        if a == "--mode" and i + 1 < len(sys.argv):
            init_mode = sys.argv[i + 1]
    node = DynamicObstacleNode(initial_mode=init_mode)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
