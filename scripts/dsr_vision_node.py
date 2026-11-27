#!/usr/bin/env python3
"""
Modular RGB-D Vision Node for Doosan M0609 + Intel RealSense D435 (Method 1: Front-tilted Gantry).

Architecture:
  1. ObjectDetectorBase / HSVColorDetector: Swappable 2D mask segmentation
     (currently OpenCV HSV; can be swapped to YOLOv8-Seg without touching 3D estimation).
  2. Pose3DEstimator: Vectorized RGB-D deprojection + TF2 transform into `base_link`,
     eliminating perspective distortion from the tilted camera view.
  3. ROS 2 Publishers:
     - /vision/target_box_pose (geometry_msgs/PoseStamped in base_link)
     - /vision/obstacle_pose   (geometry_msgs/PointStamped in base_link)
     - /vision/obstacle_info   (std_msgs/Float64MultiArray: [x, y, z_top, radius])
     - /vision/debug_image     (sensor_msgs/Image annotated with 2D/3D overlays)
"""

import json
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64MultiArray, String
import tf2_ros


@dataclass
class Detection2D:
    label: str
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    contour: np.ndarray
    area: float
    color_bgr: Tuple[int, int, int]


@dataclass
class TrackedObjectState:
    last_pose: Optional[List[float]] = None
    locked_dims: Optional[List[float]] = None  # [width_m, depth_m, height_m]
    ref_area: float = 0.0
    ref_radius: float = 0.0
    state: str = "MISSING"  # "OK", "OCCLUDED", "CARRIED", "MISSING"


class ObjectDetectorBase:
    """Abstract base class so HSV detector can be swapped for YOLOv8-Seg later."""

    def detect(self, bgr_img: np.ndarray) -> List[Detection2D]:
        raise NotImplementedError


class HSVColorDetector(ObjectDetectorBase):
    """Multi-class OpenCV HSV segmentation:
    - orange_cube (6x6x6cm)
    - green_cyl (D6x6cm)
    - obstacle_pillar (Yellow D10x25-30cm)
    """

    def __init__(self, min_area: int = 110):
        self.min_area = min_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def _extract_all(
        self, mask: np.ndarray, label: str, color_bgr: Tuple[int, int, int]
    ) -> List[Detection2D]:
        clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, self.kernel)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: List[Detection2D] = []
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
            area = float(cv2.contourArea(cnt))
            if area < self.min_area:
                continue
            obj_mask = np.zeros_like(clean)
            cv2.drawContours(obj_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            x, y, w, h = cv2.boundingRect(cnt)
            out.append(Detection2D(
                label=label, mask=obj_mask, bbox=(x, y, w, h),
                contour=cnt, area=area, color_bgr=color_bgr,
            ))
        return out

    def detect(self, bgr_img: np.ndarray) -> List[Detection2D]:
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        detections: List[Detection2D] = []

        # 1. Vivid Orange Cube Workpiece (6x6x6cm) — H=[1..20], high saturation to separate from wood table
        orange_low = cv2.inRange(hsv, np.array([1, 150, 100]), np.array([20, 255, 255]))
        orange_wrap = cv2.inRange(hsv, np.array([170, 150, 100]), np.array([180, 255, 255]))
        detections.extend(self._extract_all(cv2.bitwise_or(orange_low, orange_wrap), "orange_cube", (0, 140, 255)))

        # 2. Vivid Bright Green Cylinder Workpiece (D6x6cm) — excludes dark conveyor belt (V<105)
        green_mask = cv2.inRange(hsv, np.array([42, 130, 105]), np.array([86, 255, 255]))
        detections.extend(self._extract_all(green_mask, "green_cyl", (60, 230, 80)))

        # 3. Vivid Yellow Obstacle Pillar (D10x25-30cm)
        yellow_mask = cv2.inRange(hsv, np.array([21, 135, 120]), np.array([38, 255, 255]))
        obs_cands = self._extract_all(yellow_mask, "obstacle_pillar", (0, 230, 255))
        detections.extend(obs_cands)

        return detections


def quat_to_rot_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert ROS quaternion (x, y, z, w) to 3x3 rotation matrix."""
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def _rpy_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def compute_m0609_tcp(q: List[float]) -> np.ndarray:
    """Compute M0609 Robotiq 2F-85 TCP [x, y, z] in base_link matching NMPC FK."""
    origins = [
        ([0.0, 0.0, 0.1345], [0.0, 0.0, 0.0]),
        ([0.0, 0.0062, 0.0], [0.0, -1.571, -1.571]),
        ([0.411, 0.0, 0.0], [0.0, 0.0, 1.571]),
        ([0.0, -0.368, 0.0], [1.571, 0.0, 0.0]),
        ([0.0, 0.0, 0.0], [-1.571, 0.0, 0.0]),
        ([0.0, -0.121, 0.0], [1.571, 0.0, 0.0]),
    ]
    T = np.eye(4, dtype=np.float64)
    for idx, (pos, rpy) in enumerate(origins):
        T_orig = np.eye(4, dtype=np.float64)
        T_orig[:3, :3] = _rpy_rot(*rpy)
        T_orig[:3, 3] = pos
        c, s = math.cos(q[idx]), math.sin(q[idx])
        T_rot = np.eye(4, dtype=np.float64)
        T_rot[:2, :2] = [[c, -s], [s, c]]
        T = T @ T_orig @ T_rot
    T_tcp = np.eye(4, dtype=np.float64)
    T_tcp[2, 3] = 0.1884  # 0.0256m Doosan SVM + 0.1628m Robotiq 2F-85
    T = T @ T_tcp
    return T[:3, 3]


class Pose3DEstimator:
    """Deprojects masked pixels into camera optical frame and transforms into base_link."""

    def __init__(self, min_depth: float = 0.20, max_depth: float = 3.0):
        self.min_depth = min_depth
        self.max_depth = max_depth

    def deproject_to_base(
        self,
        mask: np.ndarray,
        depth_m: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        rot_cam_to_base: np.ndarray,
        trans_cam_to_base: np.ndarray,
    ) -> Optional[np.ndarray]:
        vs, us = np.where(mask > 0)
        if len(us) < 10:
            return None

        zs = depth_m[vs, us]
        valid = np.isfinite(zs) & (zs >= self.min_depth) & (zs <= self.max_depth)
        if np.count_nonzero(valid) < 10:
            return None

        u_v = us[valid].astype(np.float64)
        v_v = vs[valid].astype(np.float64)
        z_v = zs[valid].astype(np.float64)

        x_c = (u_v - cx) * z_v / fx
        y_c = (v_v - cy) * z_v / fy
        pts_cam = np.column_stack((x_c, y_c, z_v))  # (N, 3)

        pts_base = pts_cam @ rot_cam_to_base.T + trans_cam_to_base
        return pts_base

    @staticmethod
    def estimate_box_pose(pts_base: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
        """Return (x_c, y_c, z_grasp, yaw_rad, width_m, depth_m, height_m) in base_link."""
        z_top = float(np.percentile(pts_base[:, 2], 95))
        z_bot = float(np.percentile(pts_base[:, 2], 2))
        height_m = float(np.clip(z_top - min(0.012, z_bot), 0.02, 0.30))

        top_pts = pts_base[pts_base[:, 2] >= (z_top - 0.018)]
        if len(top_pts) < 8:
            top_pts = pts_base

        x_c = float(np.median(top_pts[:, 0]))
        y_c = float(np.median(top_pts[:, 1]))
        # Grasp center at 1/2 height of the detected object
        z_grasp = max(0.015, z_top - 0.5 * height_m)

        yaw_rad = 0.0
        width_m = float(np.clip(np.percentile(top_pts[:, 1], 97) - np.percentile(top_pts[:, 1], 3), 0.02, 0.30))
        depth_m = float(np.clip(np.percentile(top_pts[:, 0], 97) - np.percentile(top_pts[:, 0], 3), 0.02, 0.30))
        if len(top_pts) >= 12:
            xy_mm = (top_pts[:, :2] * 1000.0).astype(np.float32)
            rect = cv2.minAreaRect(xy_mm)
            angle_deg = rect[2]
            w_rect, h_rect = rect[1]
            if w_rect > 5.0 and h_rect > 5.0:
                # Map rect dimensions consistently to (width, depth)
                depth_m = float(np.clip(max(w_rect, h_rect) / 1000.0, 0.02, 0.30))
                width_m = float(np.clip(min(w_rect, h_rect) / 1000.0, 0.02, 0.30))
            if w_rect < h_rect:
                angle_deg += 90.0
            # Wrap yaw into [-45 deg, 45 deg] since box has 90-deg rotational symmetry
            while angle_deg > 45.0:
                angle_deg -= 90.0
            while angle_deg < -45.0:
                angle_deg += 90.0
            yaw_rad = math.radians(angle_deg)

        return x_c, y_c, z_grasp, yaw_rad, width_m, depth_m, height_m

    @staticmethod
    def estimate_cylinder_obstacle(pts_base: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
        """Return (x_c, y_c, z_top, radius, width_m, depth_m, height_m) compensating cylinder geometry."""
        y_min = float(np.percentile(pts_base[:, 1], 1))
        y_max = float(np.percentile(pts_base[:, 1], 99))
        # Compensate ~5% tangent edge shrinkage on rounded cylinder silhouette
        width_m = float(np.clip((y_max - y_min) * 1.06, 0.04, 0.50))
        depth_m = width_m  # Circular cylinder symmetry in XY
        radius = width_m * 0.5

        y_c = float(np.median(pts_base[:, 1]))
        # Front-facing camera sees +X side of cylinder -> shift X inward by estimated radius
        x_front = float(np.percentile(pts_base[:, 0], 96))
        x_c = x_front - radius
        z_top = float(np.percentile(pts_base[:, 2], 98))
        height_m = float(np.clip(z_top, 0.02, 0.80))
        return x_c, y_c, z_top, radius, width_m, depth_m, height_m

    def check_foreground_occlusion(
        self,
        mask: np.ndarray,
        depth_m: np.ndarray,
        bbox: Tuple[int, int, int, int],
        depth_margin: float = 0.06,
        ring_px: int = 5,
    ) -> Tuple[bool, float]:
        """Detect if a higher/closer foreground object (robot arm/gripper) overlaps the object border."""
        img_h, img_w = mask.shape[:2]
        x, y, w, h = bbox
        x1 = max(0, x - ring_px)
        y1 = max(0, y - ring_px)
        x2 = min(img_w, x + w + ring_px)
        y2 = min(img_h, y + h + ring_px)

        roi_mask = mask[y1:y2, x1:x2]
        roi_depth = depth_m[y1:y2, x1:x2]

        obj_z = roi_depth[roi_mask > 0]
        obj_z = obj_z[np.isfinite(obj_z) & (obj_z >= self.min_depth) & (obj_z <= self.max_depth)]
        if len(obj_z) < 8:
            return True, 1.0

        obj_med_depth = float(np.median(obj_z))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_px * 2 + 1, ring_px * 2 + 1))
        dilated = cv2.dilate(roi_mask, kernel)
        ring_mask = (dilated > 0) & (roi_mask == 0)

        ring_z = roi_depth[ring_mask]
        ring_z = ring_z[np.isfinite(ring_z) & (ring_z >= self.min_depth) & (ring_z <= self.max_depth)]
        if len(ring_z) < 10:
            return False, 0.0

        closer_cnt = int(np.count_nonzero(ring_z < (obj_med_depth - depth_margin)))
        fg_ratio = closer_cnt / float(len(ring_z))
        return (fg_ratio > 0.25), fg_ratio

    def is_surface_empty_at_uv(
        self,
        uv: Tuple[int, int],
        depth_m: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        rot_cb: np.ndarray,
        trans_cb: np.ndarray,
        empty_z_thresh: float = 0.032,
    ) -> bool:
        """Return True if the depth at old UV points directly to empty tray/table (no arm blocking)."""
        img_h, img_w = depth_m.shape[:2]
        u0, v0 = uv
        u1, u2 = max(0, u0 - 3), min(img_w, u0 + 4)
        v1, v2 = max(0, v0 - 3), min(img_h, v0 + 4)
        patch = depth_m[v1:v2, u1:u2]
        valid = np.isfinite(patch) & (patch >= self.min_depth) & (patch <= self.max_depth)
        if np.count_nonzero(valid) < 4:
            return False
        z_med = float(np.median(patch[valid]))
        p_cam = np.array([(u0 - cx) * z_med / fx, (v0 - cy) * z_med / fy, z_med], dtype=np.float64)
        p_base = rot_cb @ p_cam + trans_cb
        return float(p_base[2]) < empty_z_thresh


class DsrVisionNode(Node):
    def __init__(self):
        super().__init__("dsr_vision_node")

        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "d435_color_optical_frame")
        self.declare_parameter("auto_publish_nmpc_obstacle", False)

        self.base_frame = self.get_parameter("base_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value
        self.auto_pub_obs = bool(self.get_parameter("auto_publish_nmpc_obstacle").value)

        self.bridge = CvBridge()
        self.detector: ObjectDetectorBase = HSVColorDetector()
        self.estimator = Pose3DEstimator()

        # Occlusion Guard + Hold Last Valid Pose states
        self.trackers: Dict[str, TrackedObjectState] = {
            "target_box": TrackedObjectState(),
            "orange_cube": TrackedObjectState(),
            "green_cyl": TrackedObjectState(),
            "obstacle_pillar": TrackedObjectState(),
        }
        self.active_target_kind: Optional[str] = None
        self.part_tracks: Dict[int, dict] = {}
        self.next_part_id: int = 1
        self.current_q = [0.0, 0.0, 1.5708, 0.0, 1.5708, 0.0]
        self.has_joint_state = False
        self.tcp_pos = compute_m0609_tcp(self.current_q)
        self.gripper_closed = False
        self.box_carried = False
        self.box_last_uv: Optional[Tuple[int, int]] = None
        self.dyn_obs_state = None
        self.dyn_obs_ts = 0.0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cam_info: Optional[CameraInfo] = None
        self.latest_depth_m: Optional[np.ndarray] = None
        self.latest_depth_stamp = None

        # Subscribers (matching both Gazebo bridge & realsense2_camera topics)
        self.create_subscription(
            CameraInfo, "/camera/color/camera_info", self._on_camera_info, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, "/camera/aligned_depth_to_color/image_raw", self._on_depth, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, "/camera/color/image_raw", self._on_color, qos_profile_sensor_data
        )
        self.create_subscription(
            JointState, "/joint_states", self._on_joint_states, 10
        )
        self.create_subscription(
            Float64MultiArray, "/dynamic_obstacle/state", self._on_dyn_obs, 10
        )

        # Publishers
        self.box_pose_pub = self.create_publisher(PoseStamped, "/vision/target_box_pose", 10)
        self.obs_pose_pub = self.create_publisher(PointStamped, "/vision/obstacle_pose", 10)
        self.obs_info_pub = self.create_publisher(Float64MultiArray, "/vision/obstacle_info", 10)
        self.nmpc_obs_pub = self.create_publisher(PointStamped, "/nmpc/obstacle", 10)
        self.nmpc_obs_info_pub = self.create_publisher(Float64MultiArray, "/nmpc/obstacle_info", 10)
        self.debug_img_pub = self.create_publisher(Image, "/vision/debug_image", 10)
        self.status_pub = self.create_publisher(String, "/vision/status", 10)

        self.get_logger().info(
            f"DSR RGB-D Vision Node started (Occlusion Guard ON | base='{self.base_frame}')"
        )

    def _on_camera_info(self, msg: CameraInfo):
        self.cam_info = msg

    def _on_dyn_obs(self, msg: Float64MultiArray):
        if len(msg.data) >= 7:
            self.dyn_obs_state = list(msg.data[:7])
            self.dyn_obs_ts = time.monotonic()

    def _is_dynamic_obs_active(self) -> bool:
        return (
            self.dyn_obs_state is not None
            and (time.monotonic() - self.dyn_obs_ts) < 1.0
            and self.dyn_obs_state[6] > 0.5
        )

    def _on_joint_states(self, msg: JointState):
        updated_arm = False
        for i, name in enumerate(msg.name):
            if name.startswith("joint_") and len(name) == 7 and name[6].isdigit():
                idx = int(name[6]) - 1
                if 0 <= idx < 6 and i < len(msg.position):
                    self.current_q[idx] = float(msg.position[i])
                    updated_arm = True
            elif name == "robotiq_85_left_knuckle_joint" and i < len(msg.position):
                # theta > 0.16 rad means gripper width < ~68mm (actively closed / grasping)
                self.gripper_closed = float(msg.position[i]) > 0.16
                if not self.gripper_closed:
                    self.box_carried = False
        if updated_arm:
            self.has_joint_state = True
            self.tcp_pos = compute_m0609_tcp(self.current_q)

    def _publish_box_pose(self, stamp, pose: List[float]):
        sx, sy, sz, syaw = pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self.base_frame
        pose_msg.pose.position.x = sx
        pose_msg.pose.position.y = sy
        pose_msg.pose.position.z = sz
        pose_msg.pose.orientation.z = math.sin(syaw * 0.5)
        pose_msg.pose.orientation.w = math.cos(syaw * 0.5)
        self.box_pose_pub.publish(pose_msg)

    def _try_track_carried_box(self, stamp, tr_box: TrackedObjectState) -> bool:
        """If gripper is closed around the box, propagate box pose dynamically with TCP."""
        near_last = (
            tr_box.last_pose is not None
            and float(np.linalg.norm(self.tcp_pos - np.array(tr_box.last_pose[:3]))) < 0.12
        )
        if self.has_joint_state and self.gripper_closed and (self.box_carried or near_last):
            self.box_carried = True
            tr_box.state = "CARRIED"
            tr_box.last_pose = [
                float(self.tcp_pos[0]), float(self.tcp_pos[1]),
                max(0.02, float(self.tcp_pos[2]) - 0.015), float(self.current_q[5])
            ]
            self._publish_box_pose(stamp, tr_box.last_pose)
            return True
        return False

    def _on_depth(self, msg: Image):
        try:
            raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if msg.encoding in ("16UC1", "mono16") or raw.dtype == np.uint16:
                self.latest_depth_m = raw.astype(np.float32) * 0.001
            else:
                self.latest_depth_m = raw.astype(np.float32)
            self.latest_depth_stamp = msg.header.stamp
        except Exception as exc:
            self.get_logger().warn(f"Depth decode error: {exc}", throttle_duration_sec=2.0)

    def _lookup_cam_to_base(self, optical_frame: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, optical_frame, rclpy.time.Time()
            )
            t = tf.transform.translation
            q = tf.transform.rotation
            rot = quat_to_rot_matrix(q.x, q.y, q.z, q.w)
            trans = np.array([t.x, t.y, t.z], dtype=np.float64)
            return rot, trans
        except Exception as exc:
            self.get_logger().warn(
                f"Waiting for TF {self.base_frame} <- {optical_frame}: {exc}",
                throttle_duration_sec=2.0,
            )
            return None

    def _update_tracker(
        self,
        label: str,
        raw_vals: List[float],
        raw_dims: List[float],
        area: float,
        fg_occluded: bool,
        radius: Optional[float] = None,
    ) -> Tuple[str, Optional[List[float]], Optional[List[float]]]:
        """Validate against foreground/area/radius occlusion and lock clean pose + WxDxH dims."""
        tr = self.trackers[label]
        area_drop = (tr.ref_area > 0.0) and (area < 0.72 * tr.ref_area)
        rad_drop = (radius is not None) and (tr.ref_radius > 0.0) and (radius < 0.75 * tr.ref_radius)

        # If contour area is fully intact (>=85% of ref), only lock on heavy foreground overlap
        if area_drop or rad_drop or (fg_occluded and (tr.ref_area <= 0.0 or area < 0.90 * tr.ref_area)):
            tr.state = "OCCLUDED"
            return tr.state, tr.last_pose, tr.locked_dims

        # Clean, unoccluded observation -> update reference geometry and smooth pose + dims
        if tr.ref_area <= 0.0 or area > tr.ref_area:
            tr.ref_area = area
        else:
            tr.ref_area = 0.90 * tr.ref_area + 0.10 * area

        if radius is not None:
            if tr.ref_radius <= 0.0 or radius > tr.ref_radius:
                tr.ref_radius = radius
            else:
                tr.ref_radius = 0.90 * tr.ref_radius + 0.10 * radius

        if tr.last_pose is not None:
            dist_xy = math.hypot(raw_vals[0] - tr.last_pose[0], raw_vals[1] - tr.last_pose[1])
            if dist_xy < 0.05:
                alpha = 0.35
                smoothed = [
                    (1.0 - alpha) * prev + alpha * cur
                    for prev, cur in zip(tr.last_pose, raw_vals)
                ]
                if label == "target_box" and abs(raw_vals[3] - tr.last_pose[3]) > math.radians(30.0):
                    smoothed[3] = raw_vals[3]
                tr.last_pose = smoothed
            else:
                tr.last_pose = list(raw_vals)
        else:
            tr.last_pose = list(raw_vals)

        if tr.locked_dims is not None:
            beta = 0.25
            tr.locked_dims = [
                (1.0 - beta) * prev + beta * cur
                for prev, cur in zip(tr.locked_dims, raw_dims)
            ]
        else:
            tr.locked_dims = list(raw_dims)

        tr.state = "OK"
        return tr.state, tr.last_pose, tr.locked_dims

    def _on_color(self, msg: Image):
        if self.cam_info is None or self.latest_depth_m is None:
            return

        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"Color decode error: {exc}", throttle_duration_sec=2.0)
            return

        if self.latest_depth_m.shape[:2] != bgr.shape[:2]:
            return

        optical_frame = msg.header.frame_id or self.camera_frame
        tf_res = self._lookup_cam_to_base(optical_frame)
        if tf_res is None:
            return
        rot_cb, trans_cb = tf_res

        fx = float(self.cam_info.k[0])
        fy = float(self.cam_info.k[4])
        cx = float(self.cam_info.k[2])
        cy = float(self.cam_info.k[5])
        if fx <= 1e-3 or fy <= 1e-3:
            return

        detections = self.detector.detect(bgr)
        debug_img = bgr.copy()
        seen_labels = set()

        # Collect ALL visible workpieces across conveyor + board and maintain persistent sequential IDs (#1, #2, ...)
        pick_candidates = []
        conveyor_items = []
        raw_visible_parts = []
        orange_best = None
        green_best = None

        # Project conveyor Pick Boundary onto camera frame for unmistakable visual demarcation
        try:
            rot_bc = rot_cb.T
            corners_base = np.array([
                [0.275, -0.495, 0.005],
                [0.625, -0.495, 0.005],
                [0.625, -0.295, 0.005],
                [0.275, -0.295, 0.005],
            ], dtype=np.float64)
            corners_cam = (corners_base - trans_cb) @ rot_bc.T
            if np.all(corners_cam[:, 2] > 0.25):
                pts_uv = np.column_stack((
                    fx * corners_cam[:, 0] / corners_cam[:, 2] + cx,
                    fy * corners_cam[:, 1] / corners_cam[:, 2] + cy,
                )).astype(np.int32)
                cv2.polylines(debug_img, [pts_uv], isClosed=True, color=(255, 220, 40), thickness=2)
                cv2.putText(
                    debug_img, "O GAP VAT", (int(pts_uv[0][0]) + 4, max(16, int(pts_uv[0][1]) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 40), 2
                )
        except Exception:
            pass

        for det in detections:
            pts_base = self.estimator.deproject_to_base(
                det.mask, self.latest_depth_m, fx, fy, cx, cy, rot_cb, trans_cb
            )
            x, y, w, h = det.bbox
            fg_occ, fg_ratio = self.estimator.check_foreground_occlusion(
                det.mask, self.latest_depth_m, det.bbox
            )

            if pts_base is None:
                if det.label == "obstacle_pillar":
                    tr = self.trackers["obstacle_pillar"]
                    tr.state = "OCCLUDED" if tr.last_pose is not None else "MISSING"
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 140, 255), 1)
                cv2.putText(
                    debug_img, f"{det.label} (no depth)", (x, max(18, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 140, 255), 1
                )
                continue

            if det.label in ("orange_cube", "green_cyl"):
                if det.area > 3800:
                    continue  # Skip oversized flat wooden-board / background blobs
                bx, by, bz, yaw, bw, bd, bh = self.estimator.estimate_box_pose(pts_base)
                if bh < 0.025 and bz < 0.018:
                    continue  # Ignore flat surface smudges (real 6cm part has z >= 0.022m)
                if det.label == "green_cyl":
                    yaw = 0.0  # Circular cylinder symmetry
                pose_4d = [bx, by, bz, yaw]
                dims_3d = [bw, bd, bh]

                in_pick_zone = (0.305 <= bx <= 0.595) and (-0.465 <= by <= -0.325)
                in_conveyor = (0.22 <= bx <= 0.68) and (-0.95 <= by <= -0.24)
                in_workspace = (0.20 <= bx <= 0.72) and (-0.96 <= by <= 0.80)
                if in_workspace:
                    raw_visible_parts.append((
                        det.label, pose_4d, dims_3d, in_pick_zone, in_conveyor, det, fg_occ, fg_ratio
                    ))
                else:
                    cv2.drawContours(debug_img, [det.contour], -1, (140, 140, 140), 1)

                if in_conveyor:
                    conveyor_items.append({
                        "kind": det.label,
                        "pose": [round(bx, 4), round(by, 4), round(bz, 4), round(yaw, 4)],
                        "dims": [round(bw, 4), round(bd, 4), round(bh, 4)],
                        "in_zone": bool(in_pick_zone),
                    })
                    if det.label == "orange_cube":
                        if orange_best is None or (in_pick_zone and not orange_best[5]):
                            orange_best = (pose_4d, dims_3d, det, fg_occ, fg_ratio, in_pick_zone)
                    elif det.label == "green_cyl":
                        if green_best is None or (in_pick_zone and not green_best[5]):
                            green_best = (pose_4d, dims_3d, det, fg_occ, fg_ratio, in_pick_zone)
                    # STRICT PICK BOUNDARY: Only workpieces inside [TRONG O GAP] are eligible for Auto picking!
                    if in_pick_zone:
                        score = abs(by - (-0.39)) + 0.3 * abs(bx - 0.45)
                        pick_candidates.append((score, det.label, pose_4d, dims_3d, det, fg_occ, fg_ratio))

            elif det.label == "obstacle_pillar":
                if "obstacle_pillar" in seen_labels:
                    continue
                ox, oy, oz_top, orad, ow, od, oh = self.estimator.estimate_cylinder_obstacle(pts_base)
                if oz_top < 0.14 or oy < -0.26:
                    continue  # Reject short conveyor cubes misread as tall obstacle
                seen_labels.add("obstacle_pillar")
                if self._is_dynamic_obs_active():
                    dx_o, dy_o, dvx, dvy, dz_top, drad, dmode = self.dyn_obs_state
                    tr_obs = self.trackers["obstacle_pillar"]
                    tr_obs.state = "OK"
                    tr_obs.last_pose = [dx_o, dy_o, dz_top, drad]
                    tr_obs.locked_dims = [drad * 2.0, drad * 2.0, dz_top]
                    cv2.drawContours(debug_img, [det.contour], -1, det.color_bgr, 2)
                    cv2.rectangle(debug_img, (x, y), (x + w, y + h), det.color_bgr, 1)
                    tag = "DYN-CIRCLE" if int(round(dmode)) == 1 else ("DYN-ZIGZAG" if int(round(dmode)) == 2 else "RAIL-OBS")
                    txt = f"{tag} [{dx_o:.2f},{dy_o:.2f}] V=({dvx:+.2f},{dvy:+.2f})"
                    cv2.putText(
                        debug_img, txt, (x, max(18, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 230, 255), 2
                    )
                    continue

                state, info, dims = self._update_tracker(
                    "obstacle_pillar", [ox, oy, oz_top, orad], [ow, od, oh], det.area, fg_occ, radius=orad
                )
                if state == "OK" and info is not None:
                    sox, soy, soz_top, sorad = info
                    obs_msg = PointStamped()
                    obs_msg.header.stamp = msg.header.stamp
                    obs_msg.header.frame_id = self.base_frame
                    obs_msg.point.x = sox
                    obs_msg.point.y = soy
                    obs_msg.point.z = soz_top * 0.5
                    self.obs_pose_pub.publish(obs_msg)

                    info_msg = Float64MultiArray()
                    info_msg.data = [sox, soy, soz_top, sorad]
                    self.obs_info_pub.publish(info_msg)

                    if self.auto_pub_obs:
                        self.nmpc_obs_pub.publish(obs_msg)
                        self.nmpc_obs_info_pub.publish(info_msg)

                    cv2.drawContours(debug_img, [det.contour], -1, det.color_bgr, 2)
                    cv2.rectangle(debug_img, (x, y), (x + w, y + h), det.color_bgr, 1)
                    dw, dd, dh = dims if dims else (ow, od, oh)
                    txt = f"OBS-YELLOW [{sox:.3f},{soy:.3f}] {dw*100:.1f}x{dd*100:.1f}x{dh*100:.1f}cm"
                    cv2.putText(
                        debug_img, txt, (x, max(18, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 230, 255), 2
                    )
                else:
                    cv2.drawContours(debug_img, [det.contour], -1, (0, 140, 255), 2)
                    cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 140, 255), 2)
                    cv2.putText(
                        debug_img, f"OBS OCCLUDED ({fg_ratio*100:.0f}%) - LOCKED", (x, max(18, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 165, 255), 2
                    )

        # Match visible workpieces to persistent sequential IDs (#1, #2, #3...) and draw #ID overlay
        now_t = time.monotonic()
        # Sort front-to-back so initial pieces stabilize deterministically as #1, #2
        raw_visible_parts.sort(key=lambda item: -float(item[1][1]))
        matched_ids = set()
        detected_items = []
        for kind_p, pose_p, dims_p, in_pz, in_cv, det_p, fg_occ_p, _fg_r in raw_visible_parts:
            px, py = float(pose_p[0]), float(pose_p[1])
            best_pid, best_dist = None, 0.095
            for pid, tr in self.part_tracks.items():
                if pid in matched_ids or tr["kind"] != kind_p:
                    continue
                d_xy = math.hypot(px - tr["pose"][0], py - tr["pose"][1])
                if d_xy < best_dist:
                    best_dist = d_xy
                    best_pid = pid
            if best_pid is None:
                best_pid = self.next_part_id
                self.next_part_id += 1
            matched_ids.add(best_pid)
            self.part_tracks[best_pid] = {
                "kind": kind_p,
                "pose": [round(float(v), 4) for v in pose_p],
                "dims": [round(float(v), 4) for v in dims_p],
                "in_zone": bool(in_pz),
                "in_conveyor": bool(in_cv),
                "last_seen": now_t,
            }
            detected_items.append({
                "id": int(best_pid),
                "kind": kind_p,
                "pose": self.part_tracks[best_pid]["pose"],
                "dims": self.part_tracks[best_pid]["dims"],
                "in_zone": bool(in_pz),
                "in_conveyor": bool(in_cv),
            })

            x, y, w, h = det_p.bbox
            status_tag = " [TRONG O]" if in_pz else (" [BANG TAI]" if in_cv else " [BAN GO]")
            draw_col = det_p.color_bgr if in_pz else (200, 210, 220)
            cv2.drawContours(debug_img, [det_p.contour], -1, draw_col, 2)
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), draw_col, 1)
            tag_wp = "VUONG" if kind_p == "orange_cube" else "TRU"
            txt = f"#{best_pid} {tag_wp}{status_tag} [{px:.2f},{py:.2f}]"
            cv2.putText(
                debug_img, txt, (x, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, draw_col, 2
            )

        for old_pid in [pid for pid, tr in self.part_tracks.items() if (now_t - tr["last_seen"]) > 0.9]:
            del self.part_tracks[old_pid]
        if not self.part_tracks:
            self.next_part_id = 1
        detected_items.sort(key=lambda it: it["id"])

        # Update separate class trackers (orange_cube / green_cyl) for UI readouts
        for cls_name, best in (("orange_cube", orange_best), ("green_cyl", green_best)):
            tr_c = self.trackers[cls_name]
            if best is not None:
                tr_c.state = "OK"
                tr_c.last_pose = best[0]
                tr_c.locked_dims = best[1]
            else:
                tr_c.state = "MISSING"
                tr_c.last_pose = None

        # Choose active pick target on conveyor and feed target_box tracker
        if pick_candidates:
            pick_candidates.sort(key=lambda it: it[0])
            _, active_kind, pose_4d, dims_3d, det, fg_occ, fg_ratio = pick_candidates[0]
            seen_labels.add("target_box")
            if self.active_target_kind != active_kind:
                self.trackers["target_box"] = TrackedObjectState()
                self.active_target_kind = active_kind

            bx, by, bz, yaw = pose_4d
            bw, bd, bh = dims_3d
            x, y, w, h = det.bbox
            state, pose, dims = self._update_tracker(
                "target_box", pose_4d, dims_3d, det.area, fg_occ
            )
            if state == "OK" and pose is not None:
                self.box_last_uv = (int(x + w / 2), int(y + h / 2))
                if self.has_joint_state:
                    dist_tcp = float(np.linalg.norm(self.tcp_pos - np.array(pose[:3])))
                    self.box_carried = bool(self.gripper_closed and dist_tcp < 0.11)
                self._publish_box_pose(msg.header.stamp, pose)
            else:
                tr_box = self.trackers["target_box"]
                if self._try_track_carried_box(msg.header.stamp, tr_box):
                    cv2.putText(
                        debug_img, "WORKPIECE CARRIED IN GRIPPER", (x, max(18, y - 24)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 2
                    )
                else:
                    cv2.putText(
                        debug_img, f"WORKPIECE OCCLUDED ({fg_ratio*100:.0f}%) - LOCKED", (x, max(18, y - 24)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 165, 255), 2
                    )

        # Handle completely missing objects (100% occluded, carried in gripper, or removed)
        warn_y = 24
        if "target_box" not in seen_labels:
            tr_box = self.trackers["target_box"]
            if self._try_track_carried_box(msg.header.stamp, tr_box):
                cv2.putText(
                    debug_img,
                    f"CARRIED [{tr_box.last_pose[0]:.3f}, {tr_box.last_pose[1]:.3f}, {tr_box.last_pose[2]:.3f}]",
                    (12, warn_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 200, 0), 2
                )
                warn_y += 22
            elif tr_box.last_pose is not None:
                if self.box_last_uv is not None and self.estimator.is_surface_empty_at_uv(
                    self.box_last_uv, self.latest_depth_m, fx, fy, cx, cy, rot_cb, trans_cb
                ):
                    tr_box.last_pose = None
                    tr_box.locked_dims = None
                    tr_box.ref_area = 0.0
                    tr_box.state = "MISSING"
                    self.active_target_kind = None
                    self.box_last_uv = None
                else:
                    tr_box.state = "OCCLUDED"
                    cv2.putText(
                        debug_img,
                        f"TARGET 100% OCCLUDED - HOLDING [{tr_box.last_pose[0]:.3f}, {tr_box.last_pose[1]:.3f}]",
                        (12, warn_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 2
                    )
                    warn_y += 22
            else:
                tr_box.state = "MISSING"
                self.active_target_kind = None

        if "obstacle_pillar" not in seen_labels:
            tr_obs = self.trackers["obstacle_pillar"]
            if tr_obs.last_pose is not None:
                tr_obs.state = "OCCLUDED"
                cv2.putText(
                    debug_img,
                    f"OBS 100% OCCLUDED - HOLDING [{tr_obs.last_pose[0]:.3f}, {tr_obs.last_pose[1]:.3f}]",
                    (12, warn_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 2
                )
            else:
                tr_obs.state = "MISSING"

        # Publish structured status + locked 3D dimensions + ALL detected conveyor workpieces
        status_payload = {
            "box_state": self.trackers["target_box"].state,
            "obs_state": self.trackers["obstacle_pillar"].state,
            "box_pose": self.trackers["target_box"].last_pose,
            "obs_info": self.trackers["obstacle_pillar"].last_pose,
            "box_dims": self.trackers["target_box"].locked_dims,
            "obs_dims": self.trackers["obstacle_pillar"].locked_dims,
            "target_kind": self.active_target_kind,
            "orange_pose": self.trackers["orange_cube"].last_pose,
            "green_pose": self.trackers["green_cyl"].last_pose,
            "orange_in_zone": bool(orange_best[5]) if orange_best is not None else False,
            "green_in_zone": bool(green_best[5]) if green_best is not None else False,
            "conveyor_items": conveyor_items,
            "detected_items": detected_items,
        }
        status_msg = String()
        status_msg.data = json.dumps(status_payload)
        self.status_pub.publish(status_msg)

        # Crosshair at image center for quick alignment verification
        ch_x, ch_y = debug_img.shape[1] // 2, debug_img.shape[0] // 2
        cv2.drawMarker(debug_img, (ch_x, ch_y), (0, 255, 0), cv2.MARKER_CROSS, 16, 1)

        out_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
        out_msg.header = msg.header
        self.debug_img_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DsrVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
