#!/usr/bin/env python3
"""
Doosan M0609 Smart Teach Pendant v3 (DART Platform)
Advanced Graphical User Interface for ROS 2 Jazzy & Gazebo Harmonic
"""

import sys
import os
import re
import glob
import json
import math
import time
import shlex
import subprocess
import threading
from pathlib import Path

# PyQt5 imports
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer, QPointF, QRectF
from PyQt5.QtGui import QColor, QPen, QBrush, QFont, QPainter, QPainterPath

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

try:
    import gz.transport13 as gz_transport
    from gz.msgs10 import pose_pb2, boolean_pb2, double_pb2, entity_factory_pb2, entity_pb2
    HAVE_GZ_TRANSPORT = True
except ImportError:
    gz_transport = None
    pose_pb2 = None
    boolean_pb2 = None
    double_pb2 = None
    entity_factory_pb2 = None
    entity_pb2 = None
    HAVE_GZ_TRANSPORT = False

# ROS 2 imports (graceful fallback if running offline)
HAVE_ROS2 = False
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import PoseStamped, PointStamped
    from sensor_msgs.msg import JointState, Image
    from std_msgs.msg import Float64MultiArray, String, Float64, Bool
    HAVE_ROS2 = True
except ImportError:
    JointState = object
    PoseStamped = object
    PointStamped = object
    Float64MultiArray = object
    Float64 = object
    Bool = object
    String = object
    Image = object
    qos_profile_sensor_data = 10

def _resolve_programs_dir() -> str:
    """Resolve package programs directory regardless of workspace path or username."""
    for anchor in (os.path.realpath(__file__), os.path.abspath(__file__)):
        cand = Path(anchor).resolve().parent.parent / "programs"
        if cand.is_dir():
            return str(cand)
    try:
        from ament_index_python.packages import get_package_share_directory
        share_prog = os.path.join(get_package_share_directory("dsr_m0609_nmpc"), "programs")
        if os.path.isdir(share_prog):
            return share_prog
    except Exception:
        pass
    fallback = os.path.expanduser("~/.dsr_m0609_nmpc/programs")
    os.makedirs(fallback, exist_ok=True)
    return fallback


PROGRAMS_DIR = _resolve_programs_dir()
os.makedirs(PROGRAMS_DIR, exist_ok=True)


def _fix_label(lbl, fixed_h=None, align=None):
    """Lock QLabel layout size hints so live text/status updates never shake the panel."""
    if lbl is None:
        return lbl
    v_pol = QtWidgets.QSizePolicy.Fixed if fixed_h is not None else QtWidgets.QSizePolicy.Preferred
    lbl.setSizePolicy(QtWidgets.QSizePolicy.Ignored, v_pol)
    if fixed_h is not None:
        lbl.setFixedHeight(int(fixed_h))
    if align is not None:
        lbl.setAlignment(align)
    return lbl


class RosWorker(QtCore.QObject):
    """ROS 2 Node communication bridge running on Qt event loop"""
    joint_state_signal = QtCore.pyqtSignal(list)
    ee_pose_signal = QtCore.pyqtSignal(list)
    vision_image_signal = QtCore.pyqtSignal(object, bool)  # (QImage, is_debug_overlay)
    vision_box_signal = QtCore.pyqtSignal(list)            # [x, y, z, yaw_rad]
    vision_obs_signal = QtCore.pyqtSignal(list)            # [x, y, z_top, radius]
    vision_status_signal = QtCore.pyqtSignal(dict)         # occlusion status JSON
    dyn_obs_state_signal = QtCore.pyqtSignal(list)         # [x, y, vx, vy, z_top, radius, mode_id]

    def __init__(self):
        super().__init__()
        self.node = None
        self.target_pub = None
        self.target_j6_pub = None
        self.obstacle_pub = None
        self.obstacle_info_pub = None
        self.gripper_pub = None
        self.joint_cmd_pub = None
        self.dyn_mode_pub = None
        self.dyn_speed_pub = None
        self.dyn_paused_pub = None
        self.conveyor_vel_pub = None
        self._last_debug_img_ts = 0.0
        self._last_ui_img_ts = 0.0

        if HAVE_ROS2:
            try:
                if not rclpy.ok():
                    rclpy.init()
                self.node = Node('dsr_teach_pendant_gui_node')
                
                # Publishers
                self.target_pub = self.node.create_publisher(PoseStamped, '/nmpc/target_ee_pose', 10)
                self.target_j6_pub = self.node.create_publisher(Float64, '/nmpc/target_j6', 10)
                self.obstacle_pub = self.node.create_publisher(PointStamped, '/nmpc/obstacle', 10)
                self.obstacle_info_pub = self.node.create_publisher(Float64MultiArray, '/nmpc/obstacle_info', 10)
                self.gripper_pub = self.node.create_publisher(Float64MultiArray, '/gripper_position_controller/commands', 10)
                self.joint_cmd_pub = self.node.create_publisher(Float64MultiArray, '/nmpc/target_joint_pose', 10)
                self.dyn_mode_pub = self.node.create_publisher(String, '/dynamic_obstacle/mode', 10)
                self.dyn_speed_pub = self.node.create_publisher(Float64, '/dynamic_obstacle/speed', 10)
                self.dyn_paused_pub = self.node.create_publisher(Bool, '/dynamic_obstacle/paused', 10)
                self.conveyor_vel_pub = self.node.create_publisher(Float64, '/conveyor/cmd_vel', 10)

                # Subscribers
                self.node.create_subscription(JointState, '/joint_states', self._joint_cb, 10)
                self.node.create_subscription(PoseStamped, '/nmpc/current_ee_pose', self._ee_cb, 10)
                self.node.create_subscription(Image, '/vision/debug_image', self._debug_img_cb, qos_profile_sensor_data)
                self.node.create_subscription(Image, '/camera/color/image_raw', self._raw_img_cb, qos_profile_sensor_data)
                self.node.create_subscription(PoseStamped, '/vision/target_box_pose', self._vision_box_cb, 10)
                self.node.create_subscription(Float64MultiArray, '/vision/obstacle_info', self._vision_obs_cb, 10)
                self.node.create_subscription(String, '/vision/status', self._vision_status_cb, 10)
                self.node.create_subscription(Float64MultiArray, '/dynamic_obstacle/state', self._dyn_obs_cb, 10)
            except Exception as e:
                print(f"[ROS2 Warning] Initialization failed: {e}")

    def spin_once(self):
        if self.node and HAVE_ROS2:
            try:
                rclpy.spin_once(self.node, timeout_sec=0)
            except Exception:
                pass

    def _joint_cb(self, msg: JointState):
        if len(msg.position) >= 6:
            self.joint_state_signal.emit(list(msg.position[:6]))

    def _ee_cb(self, msg: PoseStamped):
        pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.ee_pose_signal.emit(pos)

    def _decode_ros_image(self, msg: Image):
        if not HAVE_NUMPY or msg.width <= 0 or msg.height <= 0:
            return None
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            expected = msg.height * msg.width * 3
            if arr.size < expected:
                return None
            img = arr[:expected].reshape((msg.height, msg.width, 3))
            if msg.encoding.lower() == 'bgr8':
                img = img[:, :, ::-1].copy()
            elif not img.flags['C_CONTIGUOUS']:
                img = np.ascontiguousarray(img)
            qimg = QtGui.QImage(
                img.data, msg.width, msg.height, msg.width * 3, QtGui.QImage.Format_RGB888
            ).copy()
            return qimg
        except Exception:
            return None

    def _debug_img_cb(self, msg: Image):
        now = time.monotonic()
        self._last_debug_img_ts = now
        if now - self._last_ui_img_ts < 0.04:  # Cap GUI preview at ~25 FPS
            return
        qimg = self._decode_ros_image(msg)
        if qimg is not None:
            self._last_ui_img_ts = now
            self.vision_image_signal.emit(qimg, True)

    def _raw_img_cb(self, msg: Image):
        now = time.monotonic()
        # Show raw D435 feed automatically whenever vision debug overlay is not active
        if now - self._last_debug_img_ts < 1.0:
            return
        if now - self._last_ui_img_ts < 0.04:
            return
        qimg = self._decode_ros_image(msg)
        if qimg is not None:
            self._last_ui_img_ts = now
            self.vision_image_signal.emit(qimg, False)

    def _vision_box_cb(self, msg: PoseStamped):
        p = msg.pose.position
        yaw = 2.0 * math.atan2(msg.pose.orientation.z, msg.pose.orientation.w)
        self.vision_box_signal.emit([float(p.x), float(p.y), float(p.z), float(yaw)])

    def _vision_obs_cb(self, msg: Float64MultiArray):
        if len(msg.data) >= 4:
            self.vision_obs_signal.emit([float(v) for v in msg.data[:4]])

    def _vision_status_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
            if isinstance(data, dict):
                self.vision_status_signal.emit(data)
        except Exception:
            pass

    def _dyn_obs_cb(self, msg: Float64MultiArray):
        if len(msg.data) >= 7:
            self.dyn_obs_state_signal.emit([float(v) for v in msg.data[:7]])

    def publish_dyn_mode(self, mode_str: str):
        if self.dyn_mode_pub and HAVE_ROS2:
            m = String()
            m.data = str(mode_str)
            self.dyn_mode_pub.publish(m)

    def publish_dyn_speed(self, scale: float):
        if self.dyn_speed_pub and HAVE_ROS2:
            m = Float64()
            m.data = float(scale)
            self.dyn_speed_pub.publish(m)

    def publish_dyn_paused(self, paused: bool):
        if self.dyn_paused_pub and HAVE_ROS2:
            m = Bool()
            m.data = bool(paused)
            self.dyn_paused_pub.publish(m)

    def publish_conveyor_vel(self, vel_mps: float):
        if self.conveyor_vel_pub and HAVE_ROS2:
            m = Float64()
            m.data = float(vel_mps)
            self.conveyor_vel_pub.publish(m)

    def publish_target(self, x, y, z, yaw=None):
        if self.target_pub and HAVE_ROS2:
            msg = PoseStamped()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            msg.pose.position.x = float(x)
            msg.pose.position.y = float(y)
            msg.pose.position.z = float(z)
            if yaw is not None:
                msg.pose.orientation.z = math.sin(float(yaw) * 0.5)
                msg.pose.orientation.w = math.cos(float(yaw) * 0.5)
                if self.target_j6_pub:
                    j6_msg = Float64()
                    j6_msg.data = float(yaw)
                    self.target_j6_pub.publish(j6_msg)
            else:
                msg.pose.orientation.w = 1.0
            self.target_pub.publish(msg)

    def publish_j6(self, j6_rad):
        if self.target_j6_pub and HAVE_ROS2:
            j6_msg = Float64()
            j6_msg.data = float(j6_rad)
            self.target_j6_pub.publish(j6_msg)

    def publish_obstacle(self, x, y, z, z_top=None, radius=None):
        if self.obstacle_pub and HAVE_ROS2:
            msg = PointStamped()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            msg.point.x = float(x)
            msg.point.y = float(y)
            msg.point.z = float(z)
            self.obstacle_pub.publish(msg)
            if self.obstacle_info_pub and (z_top is not None or radius is not None):
                info = Float64MultiArray()
                info.data = [
                    float(x),
                    float(y),
                    float(z_top if z_top is not None else max(0.25, z * 1.67)),
                    float(radius if radius is not None else 0.05),
                ]
                self.obstacle_info_pub.publish(info)

    def publish_gripper(self, width_mm):
        if self.gripper_pub and HAVE_ROS2:
            # 6-joint synchronized Robotiq 2F-85 counter-rotation
            # width: 85mm -> angle 0.0 rad, 0mm -> angle 0.8 rad
            ratio = max(0.0, min(1.0, (85.0 - width_mm) / 85.0))
            theta = ratio * 0.785
            cmd = Float64MultiArray()
            cmd.data = [theta, -theta, theta, -theta, -theta, theta]
            self.gripper_pub.publish(cmd)

    def publish_joints(self, joint_angles_rad):
        if self.joint_cmd_pub and HAVE_ROS2:
            cmd = Float64MultiArray()
            cmd.data = [float(j) for j in joint_angles_rad]
            self.joint_cmd_pub.publish(cmd)


class ContinuousJogButton(QtWidgets.QPushButton):
    """Button that continuously emits jog_signal while held down"""
    jog_signal = QtCore.pyqtSignal()

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(40) # 25 Hz jog steps
        self.timer.timeout.connect(self.on_timer)
        self.pressed.connect(self.on_press)
        self.released.connect(self.on_release)

    def on_press(self):
        self.jog_signal.emit() # immediate first step
        self.timer.start()

    def on_release(self):
        self.timer.stop()

    def on_timer(self):
        self.jog_signal.emit()


class CircularJogPad(QtWidgets.QWidget):
    """Dobot-style circular D-pad with outer X+/X-/Y-/Y+ ring and inner Z+/Z- center."""
    jog_step = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(172, 172)
        self.setMouseTracking(True)
        self.hover_zone = None
        self.pressed_zone = None
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(45)
        self.timer.timeout.connect(self._on_tick)

    def _hit_test(self, pos):
        cx, cy = self.width() * 0.5, self.height() * 0.5
        dx = pos.x() - cx
        dy = cy - pos.y()  # +dy is UP
        r = math.hypot(dx, dy)
        r_out = min(self.width(), self.height()) * 0.48
        r_gap = 33.0
        r_in = 29.0
        if r <= r_in and abs(dy) >= 2.0:
            return "zp" if dy > 0 else "zm"
        if r_gap <= r <= r_out:
            deg = math.degrees(math.atan2(dy, dx))
            if 45.0 <= deg <= 135.0:
                return "xp"
            if -135.0 <= deg <= -45.0:
                return "xm"
            if -45.0 < deg < 45.0:
                return "yp"
            return "ym"
        return None

    def _on_tick(self):
        if self.pressed_zone:
            self.jog_step.emit(self.pressed_zone)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            zone = self._hit_test(event.pos())
            if zone:
                self.pressed_zone = zone
                self.jog_step.emit(zone)
                self.timer.start()
                self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        zone = self._hit_test(event.pos())
        if zone != self.hover_zone:
            self.hover_zone = zone
            self.setCursor(Qt.PointingHandCursor if zone else Qt.ArrowCursor)
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.pressed_zone = None
        self.timer.stop()
        self.update()
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self.hover_zone = None
        self.pressed_zone = None
        self.timer.stop()
        self.update()
        super().leaveEvent(event)

    def _sector_color(self, key, is_z=False):
        if self.pressed_zone == key:
            return QColor("#0284C7")
        if self.hover_zone == key:
            return QColor("#475569") if not is_z else QColor("#3B82F6")
        return QColor("#334155") if not is_z else QColor("#1E293B")

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cx, cy = self.width() * 0.5, self.height() * 0.5
        r_out = min(self.width(), self.height()) * 0.48
        r_gap = 33.0
        r_in = 29.0

        # Subtle outer backdrop ring
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#0F172A"))
        p.drawEllipse(QPointF(cx, cy), r_out + 2.0, r_out + 2.0)

        outer_rect = QRectF(cx - r_out, cy - r_out, 2 * r_out, 2 * r_out)
        inner_rect = QRectF(cx - r_gap, cy - r_gap, 2 * r_gap, 2 * r_gap)

        sectors = [
            ("xp", 48.0, 84.0, "X+", cx, cy - (r_out + r_gap) * 0.5),
            ("ym", 138.0, 84.0, "Y-", cx - (r_out + r_gap) * 0.5, cy),
            ("xm", 228.0, 84.0, "X-", cx, cy + (r_out + r_gap) * 0.5),
            ("yp", -42.0, 84.0, "Y+", cx + (r_out + r_gap) * 0.5, cy),
        ]
        font_main = QFont("Sans", 10, QFont.Bold)
        p.setFont(font_main)

        for key, a0, span, label, tx, ty in sectors:
            path = QPainterPath()
            a0_rad = math.radians(a0)
            a1_rad = math.radians(a0 + span)
            path.moveTo(cx + r_gap * math.cos(a0_rad), cy - r_gap * math.sin(a0_rad))
            path.lineTo(cx + r_out * math.cos(a0_rad), cy - r_out * math.sin(a0_rad))
            path.arcTo(outer_rect, a0, span)
            path.lineTo(cx + r_gap * math.cos(a1_rad), cy - r_gap * math.sin(a1_rad))
            path.arcTo(inner_rect, a0 + span, -span)
            path.closeSubpath()

            p.setBrush(self._sector_color(key))
            p.setPen(QPen(QColor("#475569"), 1.0))
            p.drawPath(path)

            p.setPen(QColor("#F8FAFC"))
            p.drawText(QRectF(tx - 20, ty - 12, 40, 24), Qt.AlignCenter, label)

        # Center Z+ / Z- split halves
        in_rect = QRectF(cx - r_in, cy - r_in, 2 * r_in, 2 * r_in)
        gap_y = 2.0
        for key, start_a, span_a, label, dy_txt in [
            ("zp", 6.0, 168.0, "Z+", -r_in * 0.45),
            ("zm", 186.0, 168.0, "Z-", r_in * 0.45),
        ]:
            zp = QPainterPath()
            a0_r = math.radians(start_a)
            zp.moveTo(cx + r_in * math.cos(a0_r), cy - r_in * math.sin(a0_r) - (gap_y if key == "zp" else -gap_y))
            zp.arcTo(in_rect, start_a, span_a)
            zp.closeSubpath()
            p.setBrush(self._sector_color(key, is_z=True))
            p.setPen(QPen(QColor("#38BDF8" if self.pressed_zone == key else "#475569"), 1.0))
            p.drawPath(zp)

            p.setPen(QColor("#E0F2FE"))
            p.drawText(QRectF(cx - 20, cy + dy_txt - 10, 40, 20), Qt.AlignCenter, label)


class DobotJogPanel(QtWidgets.QFrame):
    """Synchronized Dobot-style Jog Panel: X/Y/Z/R pills + Circular D-pad + R-/R+ & Yaw tools."""
    jog_xyz_signal = QtCore.pyqtSignal(float, float, float)
    jog_r_signal = QtCore.pyqtSignal(float)
    set_r_abs_signal = QtCore.pyqtSignal(float)
    align_box_yaw_signal = QtCore.pyqtSignal()
    velocity_changed_signal = QtCore.pyqtSignal(int)

    def __init__(self, initial_vel=80, compact=False, parent=None):
        super().__init__(parent)
        self.setObjectName("cardFrame")
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Top row: Coordinate label + Velocity slider
        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(6)
        coord_lbl = QtWidgets.QLabel("🧭 Jog XYZ + R (NMPC)")
        coord_lbl.setStyleSheet("color: #38BDF8; font-weight: bold; font-size: 11px;")
        top_row.addWidget(coord_lbl)
        top_row.addStretch()

        vel_lbl = QtWidgets.QLabel("Velocity:")
        vel_lbl.setStyleSheet("color: #94A3B8; font-size: 10px;")
        self.vel_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.vel_slider.setRange(10, 100)
        self.vel_slider.setValue(int(initial_vel))
        self.vel_slider.setFixedWidth(85 if compact else 105)
        self.vel_val_lbl = QtWidgets.QLabel(f"{int(initial_vel)}%")
        self.vel_val_lbl.setStyleSheet("color: #38BDF8; font-weight: bold; font-size: 10px; min-width: 30px;")
        self.vel_slider.valueChanged.connect(self._on_vel_changed)
        top_row.addWidget(vel_lbl)
        top_row.addWidget(self.vel_slider)
        top_row.addWidget(self.vel_val_lbl)
        root.addLayout(top_row)

        # Middle area: Left (X, Y, Z, R pills) | Right (Circular D-Pad + R-/R+)
        mid = QtWidgets.QHBoxLayout()
        mid.setSpacing(10)

        left_grid = QtWidgets.QGridLayout()
        left_grid.setSpacing(5)
        self.val_pills = {}
        pill_style = (
            "background: #1E293B; color: #F8FAFC; border: 1px solid #334155; "
            "border-radius: 10px; padding: 4px 10px; font-family: monospace; "
            "font-weight: bold; font-size: 11px; min-width: 72px;"
        )
        for row_i, (axis, init_txt, color) in enumerate([
            ("X", "0.368 m", "#E2E8F0"),
            ("Y", "0.006 m", "#E2E8F0"),
            ("Z", "0.236 m", "#E2E8F0"),
            ("R", "0.0°", "#5EEAD4"),
        ]):
            lbl_a = QtWidgets.QLabel(axis)
            lbl_a.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 11px; min-width: 14px;")
            pill = QtWidgets.QLabel(init_txt)
            pill.setStyleSheet(pill_style)
            _fix_label(pill, fixed_h=24, align=Qt.AlignCenter)
            left_grid.addWidget(lbl_a, row_i, 0)
            left_grid.addWidget(pill, row_i, 1)
            self.val_pills[axis] = pill

        left_wrap = QtWidgets.QVBoxLayout()
        left_wrap.addStretch()
        left_wrap.addLayout(left_grid)
        left_wrap.addStretch()
        mid.addLayout(left_wrap)

        # Right: Circular Jog Pad + R- / R+ pills directly below
        right_col = QtWidgets.QVBoxLayout()
        right_col.setSpacing(6)
        right_col.setAlignment(Qt.AlignCenter)

        self.pad = CircularJogPad()
        self.pad.jog_step.connect(self._on_pad_step)
        right_col.addWidget(self.pad, alignment=Qt.AlignCenter)

        r_row = QtWidgets.QHBoxLayout()
        r_row.setSpacing(10)
        r_btn_style = (
            "QPushButton { background: #334155; color: #F8FAFC; border: 1px solid #475569; "
            "border-radius: 11px; padding: 4px 14px; font-weight: bold; font-size: 11px; }"
            "QPushButton:hover { background: #0D9488; border-color: #2DD4BF; }"
            "QPushButton:pressed { background: #0F766E; }"
        )
        btn_rm = ContinuousJogButton("R- ↺")
        btn_rm.setStyleSheet(r_btn_style)
        btn_rm.jog_signal.connect(lambda: self.jog_r_signal.emit(-8.0))
        btn_rp = ContinuousJogButton("R+ ↻")
        btn_rp.setStyleSheet(r_btn_style)
        btn_rp.jog_signal.connect(lambda: self.jog_r_signal.emit(8.0))
        r_row.addWidget(btn_rm)
        r_row.addWidget(btn_rp)
        right_col.addLayout(r_row)

        mid.addLayout(right_col)
        root.addLayout(mid)

        # Bottom quick rotation bar (extra R features)
        rot_tools = QtWidgets.QHBoxLayout()
        rot_tools.setSpacing(4)
        qrot_style = (
            "QPushButton { background: #1E293B; color: #CBD5E1; border: 1px solid #334155; "
            "border-radius: 5px; padding: 3px 6px; font-size: 10px; font-weight: bold; }"
            "QPushButton:hover { background: #0F766E; color: white; border-color: #14B8A6; }"
        )
        for text, slot in [
            ("R=0°", lambda: self.set_r_abs_signal.emit(0.0)),
            ("-45°", lambda: self.jog_r_signal.emit(-45.0)),
            ("+45°", lambda: self.jog_r_signal.emit(45.0)),
            ("+90°", lambda: self.jog_r_signal.emit(90.0)),
        ]:
            b = QtWidgets.QPushButton(text)
            b.setStyleSheet(qrot_style)
            b.clicked.connect(slot)
            rot_tools.addWidget(b)

        btn_align_yaw = QtWidgets.QPushButton("🎯 Khớp Yaw Hộp")
        btn_align_yaw.setStyleSheet(
            "QPushButton { background: #164E63; color: #67E8F9; border: 1px solid #0891B2; "
            "border-radius: 5px; padding: 3px 8px; font-size: 10px; font-weight: bold; }"
            "QPushButton:hover { background: #0891B2; color: white; }"
        )
        btn_align_yaw.clicked.connect(self.align_box_yaw_signal.emit)
        rot_tools.addWidget(btn_align_yaw)
        root.addLayout(rot_tools)

    def _on_vel_changed(self, val):
        self.vel_val_lbl.setText(f"{val}%")
        self.velocity_changed_signal.emit(int(val))

    def set_velocity_silent(self, val):
        self.vel_slider.blockSignals(True)
        self.vel_slider.setValue(int(val))
        self.vel_slider.blockSignals(False)
        self.vel_val_lbl.setText(f"{int(val)}%")

    def _on_pad_step(self, zone):
        step_xy = 0.015
        step_z = 0.015
        mapping = {
            "xp": (step_xy, 0.0, 0.0),
            "xm": (-step_xy, 0.0, 0.0),
            "yp": (0.0, step_xy, 0.0),
            "ym": (0.0, -step_xy, 0.0),
            "zp": (0.0, 0.0, step_z),
            "zm": (0.0, 0.0, -step_z),
        }
        if zone in mapping:
            self.jog_xyz_signal.emit(*mapping[zone])

    def update_pose(self, x, y, z, r_deg):
        self.val_pills["X"].setText(f"{x:.3f} m")
        self.val_pills["Y"].setText(f"{y:.3f} m")
        self.val_pills["Z"].setText(f"{z:.3f} m")
        self.val_pills["R"].setText(f"{r_deg:.1f}°")


class ActiveWorkspaceRadar(QtWidgets.QWidget):
    """Interactive 2D Workspace Top-Down Radar (X-Y plane with 3D obstacle projection)"""
    target_clicked = QtCore.pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(280)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMouseTracking(True)

        # Coordinate references
        self.table_x_depth = 1.2
        self.table_y_width = 0.8
        
        # State
        self.tcp_x = 0.45
        self.tcp_y = -0.25
        self.tcp_z = 0.22
        
        self.obs_x = 0.45
        self.obs_y = 0.00
        self.obs_vx = 0.00
        self.obs_vy = 0.00
        self.is_dynamic_obs = False
        self.obs_trail = []
        self.obs_z_bottom = 0.000 # table surface (robot base frame)
        self.obs_height = 0.26    # yellow pillar height (25-30cm)
        self.obs_radius = 0.05    # 5cm (10cm dia)
        self.warn_margin = 0.16   # 16cm
        self.danger_margin = 0.01 # 1cm (per user request!)
        
        self.pick_x = 0.45
        self.pick_y = -0.38
        self.place_x = 0.45
        self.place_y = 0.215

    def update_tcp(self, x, y, z):
        self.tcp_x = x
        self.tcp_y = y
        self.tcp_z = z
        self.update()

    def clear_obs_trail(self):
        self.obs_trail.clear()
        self.update()

    def update_obstacle(self, x, y, radius, warn, danger, vx=0.0, vy=0.0, is_dynamic=False):
        self.obs_x = x
        self.obs_y = y
        self.obs_vx = vx
        self.obs_vy = vy
        self.is_dynamic_obs = bool(is_dynamic)
        self.obs_radius = radius
        self.warn_margin = warn
        self.danger_margin = danger
        if is_dynamic:
            if not self.obs_trail or math.hypot(x - self.obs_trail[-1][0], y - self.obs_trail[-1][1]) > 0.008:
                self.obs_trail.append((x, y))
                if len(self.obs_trail) > 40:
                    self.obs_trail.pop(0)
        else:
            self.obs_trail.clear()
        self.update()

    def compute_3d_distance(self):
        """Accurate 3D finite cylinder clearance calculation"""
        d_xy = math.hypot(self.tcp_x - self.obs_x, self.tcp_y - self.obs_y)
        delta_r = max(0.0, d_xy - self.obs_radius)

        obs_z_top = self.obs_z_bottom + self.obs_height
        if self.tcp_z > obs_z_top:
            delta_z = self.tcp_z - obs_z_top
        elif self.tcp_z < self.obs_z_bottom:
            delta_z = self.obs_z_bottom - self.tcp_z
        else:
            delta_z = 0.0

        if d_xy <= self.obs_radius and (self.obs_z_bottom <= self.tcp_z <= obs_z_top):
            dist_3d = 0.0
        else:
            dist_3d = math.hypot(delta_r, delta_z)

        return dist_3d, delta_r, delta_z

    def world_to_screen(self, x, y):
        w = self.width()
        h = self.height()
        sx = w / 2.0 + (y / 0.88) * (w * 0.45)
        sy = h * 0.84 - (x / 0.72) * (h * 0.74)
        return sx, sy

    def screen_to_world(self, sx, sy):
        w = self.width()
        h = self.height()
        y = ((sx - w / 2.0) / (w * 0.45)) * 0.88
        x = ((h * 0.84 - sy) / (h * 0.74)) * 0.72
        return max(-0.08, min(0.70, x)), max(-0.85, min(0.85, y))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            wx, wy = self.screen_to_world(event.x(), event.y())
            self.target_clicked.emit(wx, wy)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # 1. Background grid
        p.fillRect(0, 0, w, h, QColor('#090E17'))
        p.setPen(QPen(QColor('#141E2F'), 1, Qt.DotLine))
        for gx in range(0, w, 35):
            p.drawLine(gx, 0, gx, h)
        for gy in range(0, h, 35):
            p.drawLine(0, gy, w, gy)

        # 2. Wooden Board (40cm x 110cm: x=0.25..0.65, y=-0.30..+0.80) + Conveyor (y=-0.78..-0.30)
        sx0, sy1 = self.world_to_screen(0.25, -0.30)
        sx1, sy0 = self.world_to_screen(0.65, 0.80)
        board_rect = QRectF(min(sx0, sx1), min(sy0, sy1), abs(sx1 - sx0), abs(sy1 - sy0))
        p.setPen(QPen(QColor('#B45309'), 2))
        p.setBrush(QBrush(QColor(120, 53, 15, 75)))
        p.drawRoundedRect(board_rect, 4, 4)

        cx0, cy1 = self.world_to_screen(0.25, -0.78)
        cx1, cy0 = self.world_to_screen(0.65, -0.30)
        conv_rect = QRectF(min(cx0, cx1), min(cy0, cy1), abs(cx1 - cx0), abs(cy1 - cy0))
        p.setPen(QPen(QColor('#10B981'), 1.8))
        p.setBrush(QBrush(QColor(16, 185, 129, 38)))
        p.drawRoundedRect(conv_rect, 4, 4)

        # 2b. Linear Rail Divider at y=0 (HOME) when static rail scenario is active
        if not self.is_dynamic_obs:
            rx0, ry0 = self.world_to_screen(0.25, 0.0)
            rx1, ry1 = self.world_to_screen(0.65, 0.0)
            p.setPen(QPen(QColor('#94A3B8'), 5))
            p.drawLine(QPointF(rx0, ry0), QPointF(rx1, ry1))

        # 3. Robot Base (x=0, y=0)
        bx, by = self.world_to_screen(0.0, 0.0)
        p.setPen(QPen(QColor('#FFFFFF'), 2))
        p.setBrush(QBrush(QColor('#1E293B')))
        p.drawEllipse(QPointF(bx, by), 22, 22)
        p.setFont(QFont('sans-serif', 7, QFont.Bold))
        p.setPen(QColor('#FFFFFF'))
        p.drawText(QRectF(bx - 28, by - 11, 56, 12), Qt.AlignCenter, "DOOSAN")
        p.setPen(QColor('#38BDF8'))
        p.drawText(QRectF(bx - 28, by + 1, 56, 12), Qt.AlignCenter, "M0609")

        # 4. Conveyor Scan Zone (y=-0.38)
        ax, ay = self.world_to_screen(self.pick_x, self.pick_y)
        p.setPen(QPen(QColor('#38BDF8'), 1.5, Qt.DashLine))
        p.setBrush(QBrush(QColor(56, 189, 248, 25)))
        p.drawRect(QRectF(ax - 22, ay - 22, 44, 44))
        p.setPen(QColor('#86EFAC'))
        p.setFont(QFont('sans-serif', 7, QFont.Bold))
        p.drawText(QRectF(ax - 44, ay - 36, 88, 12), Qt.AlignCenter, "BĂNG TẢI QUÉT")

        # 5. Compact 2x2 Auto Grid + Remaining Empty Manual Zone
        gx0, gy1 = self.world_to_screen(0.375, 0.140)
        gx1, gy0 = self.world_to_screen(0.525, 0.290)
        g_rect = QRectF(min(gx0, gx1), min(gy0, gy1), abs(gx1 - gx0), abs(gy1 - gy0))
        p.setPen(QPen(QColor('#EAB308'), 1.5))
        p.setBrush(QBrush(QColor(234, 179, 8, 25)))
        p.drawRect(g_rect)
        p.drawLine(QPointF(g_rect.center().x(), g_rect.top()), QPointF(g_rect.center().x(), g_rect.bottom()))
        p.drawLine(QPointF(g_rect.left(), g_rect.center().y()), QPointF(g_rect.right(), g_rect.center().y()))
        p.setPen(QColor('#FDE047'))
        p.drawText(QRectF(g_rect.left() - 10, g_rect.top() - 14, g_rect.width() + 20, 12), Qt.AlignCenter, "Ô 2×2")

        mx0, my1 = self.world_to_screen(0.27, 0.33)
        mx1, my0 = self.world_to_screen(0.63, 0.75)
        m_rect = QRectF(min(mx0, mx1), min(my0, my1), abs(mx1 - mx0), abs(my1 - my0))
        p.setPen(QPen(QColor('#38BDF8'), 1.2, Qt.DashLine))
        p.setBrush(QBrush(QColor(56, 189, 248, 18)))
        p.drawRect(m_rect)
        p.setPen(QColor('#7DD3FC'))
        p.drawText(m_rect, Qt.AlignCenter, "TRỐNG\nMANUAL")
        bx2, by2 = self.world_to_screen(self.place_x, self.place_y)

        # 6. Dynamic trail & Yellow Obstacle Pillar & Buffer Zones
        if len(self.obs_trail) >= 2:
            trail_path = QPainterPath()
            t0x, t0y = self.world_to_screen(*self.obs_trail[0])
            trail_path.moveTo(t0x, t0y)
            for tx_w, ty_w in self.obs_trail[1:]:
                psx, psy = self.world_to_screen(tx_w, ty_w)
                trail_path.lineTo(psx, psy)
            p.setPen(QPen(QColor(250, 204, 21, 150), 2.0, Qt.DotLine))
            p.setBrush(Qt.NoBrush)
            p.drawPath(trail_path)

        ox, oy = self.world_to_screen(self.obs_x, self.obs_y)
        px_per_m = (w * 0.45) / 0.88
        r_obs_px = max(6.0, self.obs_radius * px_per_m)
        r_warn_px = (self.obs_radius + self.warn_margin) * px_per_m
        r_danger_px = (self.obs_radius + self.danger_margin) * px_per_m

        # Warning buffer zone (dashed amber)
        p.setPen(QPen(QColor(245, 158, 11, 100), 1.5, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(ox, oy), r_warn_px, r_warn_px)

        # Danger cutoff zone (red)
        p.setPen(QPen(QColor(239, 68, 68, 120), 1.5))
        p.drawEllipse(QPointF(ox, oy), r_danger_px, r_danger_px)

        # Yellow obstacle pillar
        p.setPen(QPen(QColor('#FEF08A'), 2))
        p.setBrush(QBrush(QColor('#EAB308')))
        p.drawEllipse(QPointF(ox, oy), r_obs_px, r_obs_px)

        # Velocity vector arrow for dynamic obstacle
        if self.is_dynamic_obs and math.hypot(self.obs_vx, self.obs_vy) > 0.02:
            vx_e, vy_e = self.world_to_screen(self.obs_x + self.obs_vx * 0.35, self.obs_y + self.obs_vy * 0.35)
            p.setPen(QPen(QColor('#FDE047'), 2.2))
            p.drawLine(QPointF(ox, oy), QPointF(vx_e, vy_e))
            p.setBrush(QBrush(QColor('#FDE047')))
            p.drawEllipse(QPointF(vx_e, vy_e), 3.0, 3.0)

        p.setPen(QColor('#0F172A'))
        p.setFont(QFont('sans-serif', 6, QFont.Bold))
        p.drawText(QRectF(ox - 24, oy - 6, 48, 12), Qt.AlignCenter, "ĐỘNG" if self.is_dynamic_obs else "CẢN")

        # 7. NMPC Repulsion Path Curve
        path = QPainterPath()
        path.moveTo(ax, ay)
        apex_x, apex_y = self.world_to_screen(self.obs_x + 0.05, self.obs_y)
        apex_y -= (r_warn_px + 10)
        path.quadTo(apex_x, apex_y, bx2, by2)
        p.setPen(QPen(QColor('#38BDF8'), 2.5, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)

        # 8. Distance vector line from TCP to Obstacle
        tx, ty = self.world_to_screen(self.tcp_x, self.tcp_y)
        dist_3d, delta_r, delta_z = self.compute_3d_distance()
        dist_cm = dist_3d * 100.0

        if dist_cm <= (self.danger_margin * 100.0):
            line_color = QColor('#EF4444') # Red
        elif dist_cm <= (self.warn_margin * 100.0):
            line_color = QColor('#F59E0B') # Amber
        else:
            line_color = QColor('#10B981') # Green

        p.setPen(QPen(line_color, 1.5, Qt.SolidLine))
        p.drawLine(QPointF(tx, ty), QPointF(ox, oy))

        # 9. TCP Marker
        p.setPen(QPen(QColor('#FFFFFF'), 2))
        p.setBrush(QBrush(QColor('#0284C7')))
        p.drawEllipse(QPointF(tx, ty), 9, 9)
        p.setPen(QColor('#FFFFFF'))
        p.setFont(QFont('sans-serif', 7, QFont.Bold))
        p.drawText(QRectF(tx - 15, ty - 6, 30, 12), Qt.AlignCenter, "TCP")


class GripperVisualWidget(QtWidgets.QWidget):
    """Interactive 2D graphic showing Robotiq 2F-85 opening and contact pads"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(130)
        self.width_mm = 40.0

    def set_width(self, w_mm):
        self.width_mm = max(0.0, min(85.0, w_mm))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        cx = w / 2.0
        cy = h / 2.0

        # Background
        p.fillRect(0, 0, w, h, QColor('#0D1524'))
        p.setPen(QPen(QColor('#1E293B'), 1))
        p.drawRoundedRect(2, 2, w - 4, h - 4, 8, 8)

        # Gripper Palm Base (top)
        p.setPen(QPen(QColor('#475569'), 2))
        p.setBrush(QBrush(QColor('#1E293B')))
        p.drawRoundedRect(QRectF(cx - 50, cy - 45, 100, 30), 6, 6)
        p.setPen(QColor('#38BDF8'))
        p.setFont(QFont('sans-serif', 8, QFont.Bold))
        p.drawText(QRectF(cx - 50, cy - 45, 100, 30), Qt.AlignCenter, "ROBOTIQ 2F-85")

        # Jaw separation in pixels (85mm = 70px, 0mm = 16px)
        sep_px = 16 + (self.width_mm / 85.0) * 54.0

        # Left Jaw Pad (moves horizontally, stays parallel)
        left_x = cx - sep_px / 2.0 - 12
        p.setPen(QPen(QColor('#0284C7'), 2))
        p.setBrush(QBrush(QColor('#0369A1')))
        p.drawRoundedRect(QRectF(left_x, cy - 10, 12, 45), 3, 3)
        # Rubber pad lining
        p.setBrush(QBrush(QColor('#F59E0B')))
        p.drawRect(QRectF(left_x + 9, cy - 8, 3, 41))

        # Right Jaw Pad (moves horizontally, stays parallel)
        right_x = cx + sep_px / 2.0
        p.setPen(QPen(QColor('#0284C7'), 2))
        p.setBrush(QBrush(QColor('#0369A1')))
        p.drawRoundedRect(QRectF(right_x, cy - 10, 12, 45), 3, 3)
        # Rubber pad lining
        p.setBrush(QBrush(QColor('#F59E0B')))
        p.drawRect(QRectF(right_x, cy - 8, 3, 41))

        # Target Box if clamped around 35-45mm
        if 32.0 <= self.width_mm <= 48.0:
            box_w = sep_px - 4
            p.setPen(QPen(QColor('#FBBF24'), 2))
            p.setBrush(QBrush(QColor('#D97706')))
            p.drawRoundedRect(QRectF(cx - box_w/2.0, cy - 2, box_w, 30), 3, 3)
            p.setPen(QColor('#FFFFFF'))
            p.setFont(QFont('sans-serif', 7, QFont.Bold))
            p.drawText(QRectF(cx - box_w/2.0, cy - 2, box_w, 30), Qt.AlignCenter, "HỘP VÀNG")

        # Stroke dimension text
        p.setPen(QColor('#38BDF8'))
        p.setFont(QFont('monospace', 9, QFont.Bold))
        p.drawText(QRectF(cx - 60, cy + 38, 120, 16), Qt.AlignCenter, f"Khẩu độ: {self.width_mm:.1f} mm")


class GripperConfigDialog(QtWidgets.QDialog):
    """Interactive Dialog to configure Robotiq 2F-85 Gripper Stroke and Gripping Force"""
    def __init__(self, current_width=40.0, current_force=40.0, target_name="Clamp_Box", parent=None):
        super().__init__(parent)
        self.setWindowTitle("🦾 Cấu hình Lệnh Kẹp / Nhả Robotiq 2F-85")
        self.resize(480, 560)
        self.setModal(True)
        self.setStyleSheet("""
            QDialog {
                background-color: #0F172A;
                color: #F8FAFC;
                border: 1px solid #334155;
                border-radius: 10px;
            }
            QLabel {
                color: #E2E8F0;
            }
            QGroupBox {
                border: 1px solid #334155;
                border-radius: 8px;
                margin-top: 10px;
                font-weight: bold;
                font-size: 11px;
                color: #38BDF8;
                padding-top: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton {
                background-color: #1E293B;
                color: #E2E8F0;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #334155; }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        # 1. Header Banner
        h_lbl = QtWidgets.QLabel("THIẾT LẬP THÔNG SỐ TAY KẸP ROBOTIQ 2F-85")
        h_lbl.setStyleSheet("font-weight: bold; font-size: 13px; color: #38BDF8;")
        layout.addWidget(h_lbl)

        # 2. Visual Gripper Preview
        self.viz = GripperVisualWidget()
        self.viz.setFixedHeight(105)
        self.viz.set_width(current_width)
        layout.addWidget(self.viz)

        # 3. Status summary badge
        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("font-weight: bold; font-size: 11px; padding: 5px; border-radius: 4px;")
        layout.addWidget(self.lbl_status)

        # 4. Opening Width Section (0 - 85 mm)
        w_box = QtWidgets.QGroupBox("1. Khẩu độ mở mỏ kẹp (Opening Stroke)")
        w_layout = QtWidgets.QVBoxLayout(w_box)
        w_layout.setSpacing(8)

        self.lbl_w_val = QtWidgets.QLabel(f"Độ mở hiện tại: {current_width:.0f} mm")
        self.lbl_w_val.setStyleSheet("font-weight: bold; color: #FBBF24; font-family: monospace; font-size: 12px;")
        w_layout.addWidget(self.lbl_w_val)

        self.slider_w = QtWidgets.QSlider(Qt.Horizontal)
        self.slider_w.setRange(0, 85)
        self.slider_w.setValue(int(current_width))
        self.slider_w.valueChanged.connect(self.on_w_changed)
        w_layout.addWidget(self.slider_w)

        w_presets = QtWidgets.QHBoxLayout()
        btn_w_open = QtWidgets.QPushButton("Mở Hết (85mm)")
        btn_w_open.clicked.connect(lambda: self.slider_w.setValue(85))
        btn_w_box = QtWidgets.QPushButton("Kẹp Hộp (40mm)")
        btn_w_box.setStyleSheet("background: #B45309; color: white; font-weight: bold;")
        btn_w_box.clicked.connect(lambda: self.slider_w.setValue(40))
        btn_w_half = QtWidgets.QPushButton("Nửa (50mm)")
        btn_w_half.clicked.connect(lambda: self.slider_w.setValue(50))
        btn_w_close = QtWidgets.QPushButton("Đóng Hết (0mm)")
        btn_w_close.clicked.connect(lambda: self.slider_w.setValue(0))

        w_presets.addWidget(btn_w_open)
        w_presets.addWidget(btn_w_box)
        w_presets.addWidget(btn_w_half)
        w_presets.addWidget(btn_w_close)
        w_layout.addLayout(w_presets)
        layout.addWidget(w_box)

        # 5. Force Section (10 - 120 N)
        f_box = QtWidgets.QGroupBox("2. Giới hạn lực kẹp (Gripping Force)")
        f_layout = QtWidgets.QVBoxLayout(f_box)
        f_layout.setSpacing(8)

        self.lbl_f_val = QtWidgets.QLabel(f"Lực kẹp mục tiêu: {current_force:.0f} N (Bảo vệ phôi, chống bẹp)")
        self.lbl_f_val.setStyleSheet("font-weight: bold; color: #38BDF8; font-family: monospace; font-size: 12px;")
        f_layout.addWidget(self.lbl_f_val)

        self.slider_f = QtWidgets.QSlider(Qt.Horizontal)
        self.slider_f.setRange(10, 120)
        self.slider_f.setValue(int(current_force))
        self.slider_f.valueChanged.connect(self.on_f_changed)
        f_layout.addWidget(self.slider_f)

        f_presets = QtWidgets.QHBoxLayout()
        btn_f_soft = QtWidgets.QPushButton("Nhẹ (25N)")
        btn_f_soft.clicked.connect(lambda: self.slider_f.setValue(25))
        btn_f_std = QtWidgets.QPushButton("Chuẩn (50N)")
        btn_f_std.setStyleSheet("background: #0284C7; color: white; font-weight: bold;")
        btn_f_std.clicked.connect(lambda: self.slider_f.setValue(50))
        btn_f_firm = QtWidgets.QPushButton("Chặt (90N)")
        btn_f_firm.clicked.connect(lambda: self.slider_f.setValue(90))

        f_presets.addWidget(btn_f_soft)
        f_presets.addWidget(btn_f_std)
        f_presets.addWidget(btn_f_firm)
        f_layout.addLayout(f_presets)
        layout.addWidget(f_box)

        # 6. Target Name Field
        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(QtWidgets.QLabel("Tên gợi nhớ của lệnh:"))
        self.edit_name = QtWidgets.QLineEdit(target_name)
        self.edit_name.setStyleSheet("background: #1E293B; color: white; padding: 5px; border: 1px solid #334155; border-radius: 4px;")
        name_row.addWidget(self.edit_name)
        layout.addLayout(name_row)

        # 7. Action Buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_cancel = QtWidgets.QPushButton("Hủy bỏ")
        btn_cancel.clicked.connect(self.reject)

        btn_ok = QtWidgets.QPushButton("✓ Lưu Thông Số Lệnh")
        btn_ok.setStyleSheet("background: #059669; color: white; font-weight: bold; padding: 8px 20px;")
        btn_ok.clicked.connect(self.accept)

        btn_row.addWidget(btn_cancel)
        btn_row.addStretch()
        btn_row.addWidget(btn_ok)
        layout.addLayout(btn_row)

        self.update_display(int(current_width), int(current_force))

    def on_w_changed(self, val):
        self.update_display(val, self.slider_f.value())

    def on_f_changed(self, val):
        self.update_display(self.slider_w.value(), val)

    def update_display(self, w, f):
        self.viz.set_width(w)
        self.lbl_w_val.setText(f"Độ mở hiện tại: {w} mm")
        self.lbl_f_val.setText(f"Lực kẹp mục tiêu: {f} N (Bảo vệ phôi, chống bẹp)")

        if w >= 80:
            self.lbl_status.setText("🔵 TRẠNG THÁI: NHẢ KẸP HOÀN TOÀN (85mm)")
            self.lbl_status.setStyleSheet("background: #1E3A8A; color: #93C5FD; font-weight: bold; padding: 5px; border-radius: 4px;")
            if not self.edit_name.text() or "CLOSE" in self.edit_name.text() or "Clamp" in self.edit_name.text():
                self.edit_name.setText(f"OPEN_{w}MM")
        elif 32 <= w <= 48:
            self.lbl_status.setText(f"🟢 TRẠNG THÁI: KẸP CHẶT HỘP ({w}mm - Lực: {f}N)")
            self.lbl_status.setStyleSheet("background: #064E3B; color: #6EE7B7; font-weight: bold; padding: 5px; border-radius: 4px;")
            if not self.edit_name.text() or "OPEN" in self.edit_name.text():
                self.edit_name.setText(f"CLOSE_{w}MM")
        elif w <= 5:
            self.lbl_status.setText(f"⚪ TRẠNG THÁI: ĐÓNG HẾT CỠ (0mm - Lực: {f}N)")
            self.lbl_status.setStyleSheet("background: #334155; color: #CBD5E1; font-weight: bold; padding: 5px; border-radius: 4px;")
        else:
            self.lbl_status.setText(f"🟡 TRẠNG THÁI: KHẨU ĐỘ TÙY CHỈNH ({w}mm - Lực: {f}N)")
            self.lbl_status.setStyleSheet("background: #78350F; color: #FDE68A; font-weight: bold; padding: 5px; border-radius: 4px;")

    def get_result(self):
        w = self.slider_w.value()
        f = self.slider_f.value()
        name = self.edit_name.text().strip() or (f"CLOSE_{w}MM" if w < 75 else f"OPEN_{w}MM")
        
        if w >= 80:
            desc = f"Nhả mỏ kẹp ra hoàn toàn (khẩu độ {w}mm)"
        elif 32 <= w <= 48:
            desc = f"Kẹp chặt hộp vàng song song ({w}mm, Lực: {f}N)"
        elif w <= 5:
            desc = f"Đóng hết mỏ kẹp (0mm, Lực: {f}N)"
        else:
            desc = f"Thiết lập khẩu độ kẹp {w}mm với lực {f}N"

        params = f"{w}mm | {f}N"
        return name, params, desc, w, f


class DelayConfigDialog(QtWidgets.QDialog):
    """Interactive Dialog to configure Delay/Wait duration"""
    def __init__(self, current_delay=1.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⏱ Cấu hình Thời Gian Delay / Tạm Dừng")
        self.resize(380, 260)
        self.setModal(True)
        self.setStyleSheet("""
            QDialog {
                background-color: #0F172A;
                color: #F8FAFC;
                border: 1px solid #334155;
                border-radius: 10px;
            }
            QLabel { color: #E2E8F0; }
            QPushButton {
                background-color: #1E293B;
                color: #E2E8F0;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #334155; }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        h_lbl = QtWidgets.QLabel("THIẾT LẬP THỜI GIAN TẠM DỪNG (DELAY)")
        h_lbl.setStyleSheet("font-weight: bold; font-size: 12px; color: #38BDF8;")
        layout.addWidget(h_lbl)

        self.lbl_val = QtWidgets.QLabel(f"Thời gian chờ: {current_delay:.1f} giây")
        self.lbl_val.setStyleSheet("font-weight: bold; font-size: 14px; color: #34D399; font-family: monospace;")
        layout.addWidget(self.lbl_val)

        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setRange(1, 100) # 0.1s to 10.0s
        self.slider.setValue(int(current_delay * 10))
        self.slider.valueChanged.connect(self.on_changed)
        layout.addWidget(self.slider)

        presets = QtWidgets.QHBoxLayout()
        for t_val, label in [(0.5, "0.5s"), (1.0, "1.0s (Chuẩn)"), (2.0, "2.0s"), (3.0, "3.0s")]:
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(lambda _, v=t_val: self.slider.setValue(int(v * 10)))
            presets.addWidget(btn)
        layout.addLayout(presets)

        desc_lbl = QtWidgets.QLabel("Mẹo: Dùng delay 0.8s - 1.5s sau khi kẹp để mỏ kẹp ngậm chặt phôi ổn định.")
        desc_lbl.setStyleSheet("color: #94A3B8; font-size: 10px; font-style: italic;")
        layout.addWidget(desc_lbl)

        btn_row = QtWidgets.QHBoxLayout()
        btn_cancel = QtWidgets.QPushButton("Hủy bỏ")
        btn_cancel.clicked.connect(self.reject)

        btn_ok = QtWidgets.QPushButton("✓ Lưu Thời Gian Delay")
        btn_ok.setStyleSheet("background: #059669; color: white; font-weight: bold; padding: 8px 18px;")
        btn_ok.clicked.connect(self.accept)

        btn_row.addWidget(btn_cancel)
        btn_row.addStretch()
        btn_row.addWidget(btn_ok)
        layout.addLayout(btn_row)

    def on_changed(self, val):
        secs = val / 10.0
        self.lbl_val.setText(f"Thời gian chờ: {secs:.1f} giây")

    def get_result(self):
        secs = self.slider.value() / 10.0
        target = f"DELAY_{secs:.1f}S"
        params = f"{secs:.1f}s"
        desc = f"Tạm dừng robot {secs:.1f} giây để ổn định"
        return target, params, desc, secs


class OrbitVisualWidget(QtWidgets.QWidget):
    """Visual preview of orbit trajectory with center, radius, swept angle and direction"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(130)
        self.radius = 0.14 # meters
        self.speed = 60.0 # deg/s
        self.turns = 1.0 # revolutions
        self.center_x = 0.45
        self.center_y = 0.0

    def set_params(self, r, s, t, cx=0.45, cy=0.0):
        self.radius = float(r)
        self.speed = float(s)
        self.turns = float(t)
        self.center_x = float(cx)
        self.center_y = float(cy)
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        w = self.width()
        h = self.height()

        # Dark sleek background
        bg_color = QtGui.QColor("#090D16")
        painter.fillRect(0, 0, w, h, bg_color)
        painter.setPen(QtGui.QColor("#334155"))
        painter.drawRoundedRect(0, 0, w - 1, h - 1, 6, 6)

        center_px = QtCore.QPointF(w * 0.38, h * 0.5)

        # Scale factor: map radius 0.05m..0.30m to pixels 25px..52px
        r_clamped = max(0.05, min(0.30, self.radius))
        r_px = 25.0 + (r_clamped - 0.05) / 0.25 * 30.0

        # Draw grid concentric rings (reference radar style)
        painter.setPen(QtGui.QPen(QtGui.QColor("#1E293B"), 1, Qt.DotLine))
        painter.drawEllipse(center_px, r_px * 0.6, r_px * 0.6)
        painter.drawEllipse(center_px, r_px * 1.3, r_px * 1.3)

        # Draw Center Obstacle (red glowing pillar)
        painter.setBrush(QtGui.QColor(239, 68, 68, 200))
        painter.setPen(QtGui.QPen(QtGui.QColor("#EF4444"), 2))
        painter.drawEllipse(center_px, 9, 9)

        # Center label
        painter.setFont(QtGui.QFont("DejaVu Sans", 8, QtGui.QFont.Bold))
        painter.setPen(QtGui.QColor("#FCA5A5"))
        painter.drawText(int(center_px.x()) - 18, int(center_px.y()) - 13, "TÂM CỘT")

        # Draw circular orbit path (dashed purple line)
        painter.setBrush(Qt.NoBrush)
        orbit_pen = QtGui.QPen(QtGui.QColor("#7C3AED"), 2, Qt.DashLine)
        painter.setPen(orbit_pen)
        painter.drawEllipse(center_px, r_px, r_px)

        # Draw actual swept arc path based on turns
        sweep_deg = self.turns * 360.0
        active_pen = QtGui.QPen(QtGui.QColor("#C084FC"), 3)
        painter.setPen(active_pen)
        span_16th = int(-min(360.0, sweep_deg) * 16)
        rect = QtCore.QRectF(center_px.x() - r_px, center_px.y() - r_px, r_px * 2, r_px * 2)
        painter.drawArc(rect, 0, span_16th)

        # Direction indicator dot
        arrow_ang_rad = math.radians(-45)
        arrow_tip_x = center_px.x() + r_px * math.cos(arrow_ang_rad)
        arrow_tip_y = center_px.y() + r_px * math.sin(arrow_ang_rad)
        painter.setBrush(QtGui.QColor("#38BDF8"))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QtCore.QPointF(arrow_tip_x, arrow_tip_y), 4, 4)

        # Dimension line for Radius R
        painter.setPen(QtGui.QPen(QtGui.QColor("#FBBF24"), 1, Qt.SolidLine))
        painter.drawLine(int(center_px.x()), int(center_px.y()), int(center_px.x() + r_px), int(center_px.y()))
        painter.setFont(QtGui.QFont("DejaVu Sans", 8, QtGui.QFont.Bold))
        painter.setPen(QtGui.QColor("#FDE68A"))
        painter.drawText(int(center_px.x() + 4), int(center_px.y()) + 14, f"R={self.radius*100:.0f}cm")

        # Telemetry info box on right side
        info_x = int(w * 0.65)
        painter.setFont(QtGui.QFont("DejaVu Sans", 8))
        painter.setPen(QtGui.QColor("#94A3B8"))
        painter.drawText(info_x, 26, "Bán kính quỹ đạo:")
        painter.setFont(QtGui.QFont("DejaVu Sans", 9, QtGui.QFont.Bold))
        painter.setPen(QtGui.QColor("#38BDF8"))
        painter.drawText(info_x, 41, f"{self.radius*100:.0f} cm ({self.radius:.2f} m)")

        painter.setFont(QtGui.QFont("DejaVu Sans", 8))
        painter.setPen(QtGui.QColor("#94A3B8"))
        painter.drawText(info_x, 61, "Tốc độ quay:")
        painter.setFont(QtGui.QFont("DejaVu Sans", 9, QtGui.QFont.Bold))
        painter.setPen(QtGui.QColor("#34D399"))
        est_sec = (self.turns * 360.0) / max(1.0, self.speed)
        painter.drawText(info_x, 76, f"{self.speed:.0f} °/s (~{est_sec:.1f}s)")

        painter.setFont(QtGui.QFont("DejaVu Sans", 8))
        painter.setPen(QtGui.QColor("#94A3B8"))
        painter.drawText(info_x, 96, "Số vòng & Tổng góc:")
        painter.setFont(QtGui.QFont("DejaVu Sans", 9, QtGui.QFont.Bold))
        painter.setPen(QtGui.QColor("#F472B6"))
        painter.drawText(info_x, 111, f"{self.turns:.2f} vòng ({sweep_deg:.0f}°)")


class OrbitConfigDialog(QtWidgets.QDialog):
    """Interactive Dialog to configure Orbit / RotateCircle parameters"""
    def __init__(self, current_radius=0.14, current_speed=60.0, current_turns=1.0,
                 center_x=0.45, center_y=0.0, target_name="Full_360_Orbit", parent=None):
        super().__init__(parent)
        self.setWindowTitle("🌀 Cấu hình Quỹ Đạo Tròn (RotateCircle / Orbit)")
        self.resize(500, 600)
        self.setModal(True)
        self.setStyleSheet("""
            QDialog {
                background-color: #0F172A;
                color: #F8FAFC;
                border: 1px solid #334155;
                border-radius: 10px;
            }
            QLabel {
                color: #E2E8F0;
            }
            QGroupBox {
                border: 1px solid #334155;
                border-radius: 8px;
                margin-top: 10px;
                font-weight: bold;
                font-size: 11px;
                color: #C084FC;
                padding-top: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton {
                background-color: #1E293B;
                color: #E2E8F0;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #334155; }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        # 1. Header Banner
        h_lbl = QtWidgets.QLabel("THIẾT LẬP THÔNG SỐ QUỸ ĐẠO TRÒN (ROTATE CIRCLE)")
        h_lbl.setStyleSheet("font-weight: bold; font-size: 13px; color: #C084FC;")
        layout.addWidget(h_lbl)

        # 2. Visual Orbit Preview
        self.viz = OrbitVisualWidget()
        self.viz.set_params(current_radius, current_speed, current_turns, center_x, center_y)
        layout.addWidget(self.viz)

        # 3. Radius Section (5cm - 30cm)
        r_box = QtWidgets.QGroupBox("1. Bán kính quỹ đạo tròn (Orbit Radius)")
        r_layout = QtWidgets.QVBoxLayout(r_box)
        r_layout.setSpacing(6)

        self.lbl_r_val = QtWidgets.QLabel(f"Bán kính: {current_radius*100:.0f} cm ({current_radius:.2f} m)")
        self.lbl_r_val.setStyleSheet("font-weight: bold; color: #38BDF8; font-family: monospace; font-size: 12px;")
        r_layout.addWidget(self.lbl_r_val)

        self.slider_r = QtWidgets.QSlider(Qt.Horizontal)
        self.slider_r.setRange(5, 30) # 5cm to 30cm
        self.slider_r.setValue(int(round(current_radius * 100)))
        self.slider_r.valueChanged.connect(self.on_param_changed)
        r_layout.addWidget(self.slider_r)

        r_presets = QtWidgets.QHBoxLayout()
        for r_val, label in [(10, "10cm (Gần)"), (14, "14cm (Chuẩn)"), (18, "18cm (Rộng)"), (25, "25cm (Xa)")]:
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(lambda _, v=r_val: self.slider_r.setValue(v))
            r_presets.addWidget(btn)
        r_layout.addLayout(r_presets)
        layout.addWidget(r_box)

        # 4. Speed Section (15 deg/s - 180 deg/s)
        s_box = QtWidgets.QGroupBox("2. Tốc độ quay (Orbit Angular Speed)")
        s_layout = QtWidgets.QVBoxLayout(s_box)
        s_layout.setSpacing(6)

        self.lbl_s_val = QtWidgets.QLabel(f"Tốc độ quay: {current_speed:.0f} °/s")
        self.lbl_s_val.setStyleSheet("font-weight: bold; color: #34D399; font-family: monospace; font-size: 12px;")
        s_layout.addWidget(self.lbl_s_val)

        self.slider_s = QtWidgets.QSlider(Qt.Horizontal)
        self.slider_s.setRange(15, 180) # 15°/s to 180°/s
        self.slider_s.setValue(int(round(current_speed)))
        self.slider_s.valueChanged.connect(self.on_param_changed)
        s_layout.addWidget(self.slider_s)

        s_presets = QtWidgets.QHBoxLayout()
        for s_val, label in [(30, "Chậm (30°/s)"), (60, "Chuẩn (60°/s)"), (90, "Nhanh (90°/s)"), (120, "Rất Nhanh (120°/s)")]:
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(lambda _, v=s_val: self.slider_s.setValue(v))
            s_presets.addWidget(btn)
        s_layout.addLayout(s_presets)
        layout.addWidget(s_box)

        # 5. Number of Revolutions / Turns (0.25 to 5.0 turns)
        t_box = QtWidgets.QGroupBox("3. Số vòng quay (Revolutions / Total Angle)")
        t_layout = QtWidgets.QVBoxLayout(t_box)
        t_layout.setSpacing(6)

        total_deg = current_turns * 360.0
        self.lbl_t_val = QtWidgets.QLabel(f"Số vòng: {current_turns:.2f} vòng ({total_deg:.0f}°)")
        self.lbl_t_val.setStyleSheet("font-weight: bold; color: #F472B6; font-family: monospace; font-size: 12px;")
        t_layout.addWidget(self.lbl_t_val)

        self.slider_t = QtWidgets.QSlider(Qt.Horizontal)
        self.slider_t.setRange(1, 20) # 1 * 0.25 = 0.25 to 20 * 0.25 = 5.0 turns
        self.slider_t.setValue(int(round(current_turns * 4)))
        self.slider_t.valueChanged.connect(self.on_param_changed)
        t_layout.addWidget(self.slider_t)

        t_presets = QtWidgets.QHBoxLayout()
        for t_val, label in [(0.5, "0.5 Vòng (180°)"), (1.0, "1.0 Vòng (360°)"), (2.0, "2.0 Vòng (720°)"), (3.0, "3.0 Vòng (1080°)")]:
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(lambda _, v=t_val: self.slider_t.setValue(int(round(v * 4))))
            t_presets.addWidget(btn)
        t_layout.addLayout(t_presets)
        layout.addWidget(t_box)

        # 6. Center Coordinates & Target Name
        opt_box = QtWidgets.QGroupBox("4. Tâm xoay & Tên lệnh")
        opt_layout = QtWidgets.QHBoxLayout(opt_box)
        opt_layout.setSpacing(10)

        opt_layout.addWidget(QtWidgets.QLabel("Tâm X(m):"))
        self.sp_cx = QtWidgets.QDoubleSpinBox()
        self.sp_cx.setRange(-1.0, 1.0)
        self.sp_cx.setSingleStep(0.01)
        self.sp_cx.setValue(center_x)
        self.sp_cx.setStyleSheet("background: #1E293B; color: white; padding: 3px;")
        self.sp_cx.valueChanged.connect(self.on_param_changed)
        opt_layout.addWidget(self.sp_cx)

        opt_layout.addWidget(QtWidgets.QLabel("Tâm Y(m):"))
        self.sp_cy = QtWidgets.QDoubleSpinBox()
        self.sp_cy.setRange(-1.0, 1.0)
        self.sp_cy.setSingleStep(0.01)
        self.sp_cy.setValue(center_y)
        self.sp_cy.setStyleSheet("background: #1E293B; color: white; padding: 3px;")
        self.sp_cy.valueChanged.connect(self.on_param_changed)
        opt_layout.addWidget(self.sp_cy)

        opt_layout.addWidget(QtWidgets.QLabel("Tên gợi nhớ:"))
        self.edit_name = QtWidgets.QLineEdit(target_name)
        self.edit_name.setStyleSheet("background: #1E293B; color: white; padding: 4px; border: 1px solid #334155; border-radius: 4px;")
        opt_layout.addWidget(self.edit_name)

        layout.addWidget(opt_box)

        # 7. Action Buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_cancel = QtWidgets.QPushButton("Hủy bỏ")
        btn_cancel.clicked.connect(self.reject)

        btn_ok = QtWidgets.QPushButton("✓ Lưu Thông Số Quỹ Đạo")
        btn_ok.setStyleSheet("background: #7C3AED; color: white; font-weight: bold; padding: 8px 20px;")
        btn_ok.clicked.connect(self.accept)

        btn_row.addWidget(btn_cancel)
        btn_row.addStretch()
        btn_row.addWidget(btn_ok)
        layout.addLayout(btn_row)

        self.on_param_changed()

    def on_param_changed(self):
        r = self.slider_r.value() / 100.0
        s = float(self.slider_s.value())
        turns = self.slider_t.value() * 0.25
        cx = self.sp_cx.value()
        cy = self.sp_cy.value()
        total_deg = turns * 360.0
        est_sec = total_deg / max(1.0, s)

        self.lbl_r_val.setText(f"Bán kính: {r*100:.0f} cm ({r:.2f} m)")
        self.lbl_s_val.setText(f"Tốc độ quay: {s:.0f} °/s (~{est_sec:.1f}s/chu kỳ)")
        self.lbl_t_val.setText(f"Số vòng: {turns:.2f} vòng ({total_deg:.0f}°)")

        # Auto-suggest target name if user hasn't typed custom name
        cur_name = self.edit_name.text().strip()
        if cur_name in ["", "Full_360_Orbit", "Half_180_Orbit", "Double_720_Orbit", "Triple_1080_Orbit"] or cur_name.startswith("Orbit_"):
            if turns == 0.5:
                self.edit_name.setText("Half_180_Orbit")
            elif turns == 1.0:
                self.edit_name.setText("Full_360_Orbit")
            elif turns == 2.0:
                self.edit_name.setText("Double_720_Orbit")
            elif turns == 3.0:
                self.edit_name.setText("Triple_1080_Orbit")
            else:
                self.edit_name.setText(f"Orbit_{turns:.2f}T")

        self.viz.set_params(r, s, turns, cx, cy)

    def get_config(self):
        r = self.slider_r.value() / 100.0
        s = float(self.slider_s.value())
        turns = self.slider_t.value() * 0.25
        cx = self.sp_cx.value()
        cy = self.sp_cy.value()
        total_deg = turns * 360.0

        name = self.edit_name.text().strip()
        if not name:
            if turns == 0.5:
                name = "Half_180_Orbit"
            elif turns == 1.0:
                name = "Full_360_Orbit"
            elif turns == 2.0:
                name = "Double_720_Orbit"
            else:
                name = f"Orbit_{turns:.1f}T"

        desc = f"Quay {turns:.2f} vòng ({total_deg:.0f}°) quanh tâm [{cx:.2f}, {cy:.2f}], R={r*100:.0f}cm, V={s:.0f}°/s"
        params = f"center:[{cx:.2f}, {cy:.2f}], radius:{r:.2f}, speed:{s:.1f}, turns:{turns:.2f}, angle:{total_deg:.0f}"
        return name, params, desc, r, s, turns, [cx, cy]



class InstructionCardWidget(QtWidgets.QFrame):
    """Custom styled card for Program Tree items (matches HTML mockup styling)"""
    def __init__(self, step_idx, step_data, on_up, on_down, on_delete, on_edit=None, parent=None):
        super().__init__(parent)
        self.step_idx = step_idx
        self.on_edit = on_edit
        self.setObjectName("instructionCard")
        self.setStyleSheet("""
            QFrame#instructionCard {
                background: #121B2C;
                border: 1px solid #2B3D5E;
                border-radius: 8px;
                padding: 4px;
            }
            QFrame#instructionCard:hover {
                border: 1px solid #38BDF8;
                background: #162238;
            }
        """)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(10)

        # Number circle badge
        num_lbl = QtWidgets.QLabel(str(step_idx + 1))
        num_lbl.setFixedSize(22, 22)
        num_lbl.setAlignment(Qt.AlignCenter)
        num_lbl.setStyleSheet("background: #1E293B; color: #94A3B8; border-radius: 11px; font-weight: bold; font-size: 10px;")
        layout.addWidget(num_lbl)

        # Type badge with distinct colors
        t = step_data.get('type', 'MoveNMPC')
        type_lbl = QtWidgets.QLabel(t)
        type_lbl.setFixedWidth(82)
        type_lbl.setAlignment(Qt.AlignCenter)
        if t == 'MoveNMPC':
            type_lbl.setStyleSheet("background: #0C4A6E; color: #38BDF8; border: 1px solid #0284C7; border-radius: 4px; font-weight: bold; font-size: 10px; font-family: monospace;")
        elif t == 'RotateCircle':
            type_lbl.setStyleSheet("background: #3B0764; color: #C084FC; border: 1px solid #7C3AED; border-radius: 4px; font-weight: bold; font-size: 10px; font-family: monospace;")
        elif t == 'MoveJ':
            type_lbl.setStyleSheet("background: #064E3B; color: #34D399; border: 1px solid #059669; border-radius: 4px; font-weight: bold; font-size: 10px; font-family: monospace;")
        elif t == 'Gripper':
            type_lbl.setStyleSheet("background: #451A03; color: #FBBF24; border: 1px solid #D97706; border-radius: 4px; font-weight: bold; font-size: 10px; font-family: monospace;")
        else:
            type_lbl.setStyleSheet("background: #1E293B; color: #CBD5E1; border: 1px solid #475569; border-radius: 4px; font-weight: bold; font-size: 10px; font-family: monospace;")
        layout.addWidget(type_lbl)

        # Target name & Params
        info_layout = QtWidgets.QVBoxLayout()
        info_layout.setSpacing(1)
        
        top_row = QtWidgets.QHBoxLayout()
        tg_lbl = QtWidgets.QLabel(step_data.get('target', 'Target'))
        tg_lbl.setStyleSheet("font-weight: bold; color: #FFFFFF; font-size: 11px;")
        param_lbl = QtWidgets.QLabel(str(step_data.get('params', '')))
        param_lbl.setStyleSheet("color: #38BDF8; font-family: monospace; font-size: 11px; margin-left: 6px;")
        top_row.addWidget(tg_lbl)
        top_row.addWidget(param_lbl)
        top_row.addStretch()

        desc_lbl = QtWidgets.QLabel(step_data.get('desc', ''))
        desc_lbl.setStyleSheet("color: #94A3B8; font-size: 10px;")

        info_layout.addLayout(top_row)
        info_layout.addWidget(desc_lbl)
        layout.addLayout(info_layout, stretch=1)

        # Action Buttons
        if on_edit:
            btn_edit = QtWidgets.QPushButton("✎")
            btn_edit.setFixedSize(22, 22)
            btn_edit.setStyleSheet("padding: 0px; font-size: 11px; background: #0284C7; color: white;")
            btn_edit.setToolTip("Chỉnh sửa thông số bước này (hoặc nhấp đúp thẻ)")
            btn_edit.clicked.connect(lambda: on_edit(step_idx))
            layout.addWidget(btn_edit)

        btn_up = QtWidgets.QPushButton("▲")
        btn_up.setFixedSize(22, 22)
        btn_up.setStyleSheet("padding: 0px; font-size: 9px;")
        btn_up.clicked.connect(lambda: on_up(step_idx))

        btn_down = QtWidgets.QPushButton("▼")
        btn_down.setFixedSize(22, 22)
        btn_down.setStyleSheet("padding: 0px; font-size: 9px;")
        btn_down.clicked.connect(lambda: on_down(step_idx))

        btn_del = QtWidgets.QPushButton("✕")
        btn_del.setFixedSize(22, 22)
        btn_del.setStyleSheet("padding: 0px; font-size: 10px; background: #7F1D1D; color: #FCA5A5;")
        btn_del.clicked.connect(lambda: on_delete(step_idx))

        layout.addWidget(btn_up)
        layout.addWidget(btn_down)
        layout.addWidget(btn_del)

    def mouseDoubleClickEvent(self, event):
        if self.on_edit:
            self.on_edit(self.step_idx)


class TeachPendantApp(QtWidgets.QMainWindow):
    """Doosan M0609 Smart Teach Pendant Main Window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Doosan M0609 Teach Pendant (DART Platform v3.2)")
        self.resize(1260, 820)
        self.setMinimumSize(940, 680)

        # Background process handles
        self.processes = {}

        # ROS Bridge Node
        self.ros = RosWorker()
        self.ros.joint_state_signal.connect(self.on_joint_state)
        self.ros.ee_pose_signal.connect(self.on_ee_pose)
        self.ros.vision_image_signal.connect(self.on_vision_image)
        self.ros.vision_box_signal.connect(self.on_vision_box)
        self.ros.vision_obs_signal.connect(self.on_vision_obs)
        self.ros.vision_status_signal.connect(self.on_vision_status)
        self.ros.dyn_obs_state_signal.connect(self.on_dyn_obs_state)

        # Simulation Problem & Dynamic Obstacle State ("rail" default = Bản 1, "dynamic" = Bản 2)
        self.sim_scenario = "rail"     # "rail" | "dynamic"
        self.dyn_obs_mode = "circle"   # "circle" | "zigzag" | "rail"
        self.dyn_obs_speed = 1.0
        self.dyn_obs_paused = False
        self.running_gz_world = None

        # Vision Telemetry State
        self.vision_box_pose = None   # [x, y, z, yaw_rad]
        self.vision_box_ts = 0.0
        self.vision_obs_info = None   # [x, y, z_top, radius]
        self.vision_obs_ts = 0.0
        self.vision_sync_obs = False

        # Execution Engine State
        self.current_program_name = "Pick_and_Place_NMPC"
        self.program_steps = []
        self.current_step_idx = 0
        self.is_running = False
        self.is_paused = False
        self.is_auto_mode = True # Default AUTO mode
        self.step_in_progress = False
        self.step_timer = 0.0
        self.settle_timer = 0.0
        self.cycle_time = 0.0
        self.current_target_xyz = [0.450, -0.250, 0.220]
        self.gripper_force = 40.0 # Force limit in Newtons
        self.orbit_phase_angle = 0.0
        self.orbit_start_angle = 0.0
        self.orbit_traveled_angle = 0.0
        self.orbit_speed = 60.0
        self.orbit_center = [0.45, 0.0, 0.25]
        self.orbit_radius = 0.14
        self.orbit_target_angle = 360.0
        self.wait_duration = 1.0

        # Robot Telemetry
        self.joint_angles = [0.0] * 6
        self.tcp_pos = [0.450, -0.250, 0.220]
        self.desired_j6_rad = 0.0
        self.gripper_width = 40.0
        self.jog_panels = []

        # Vision Pick & Place Tab 3 State (Auto 2x2 Continuous vs Manual 1-Shot)
        self.selected_vision_target = "target_box"
        self.vpp_manual_target_pref = "auto"  # "auto" | "orange_cube" | "green_cyl"
        self.place_pose_taught = None   # [x, y, z, j6_rad]
        self.vpp_run_mode = "auto"      # "auto" | "manual"
        self.vpp_auto_armed = False
        self.vpp_conveyor_autofeed = True
        self.conveyor_running = False
        self.conveyor_direction = 1     # +1 = Tiến (+Y), -1 = Lùi (-Y)
        self.conveyor_speed_mps = 0.09
        self._conv_last_ts = 0.0
        self._conv_last_pub_ts = 0.0
        self.spawned_workpiece_names = []
        self._spawn_counter = 0
        self.vpp_dwell_sec = 3.0
        self._last_vpp_phase_msg = ""
        self.gz_node = gz_transport.Node() if HAVE_GZ_TRANSPORT else None
        self._gz_conv_pub = (
            self.gz_node.advertise("/model/conveyor_feeder/link/conveyor_link/track_cmd_vel", double_pb2.Double)
            if (self.gz_node is not None and double_pb2 is not None) else None
        )
        self.vpp_active_target_kind = None  # "orange_cube" | "green_cyl"
        self.vpp_orange_pose = None
        self.vpp_green_pose = None
        self.vpp_orange_in_zone = False
        self.vpp_green_in_zone = False
        self.vpp_conveyor_items = []
        self.vpp_auto_orange_idx = 0
        self.vpp_auto_green_idx = 0
        self.vpp_auto_slots = {
            "orange_cube": [(0.390, 0.185, 0.028), (0.390, 0.305, 0.028)],
            "green_cyl":   [(0.510, 0.185, 0.028), (0.510, 0.305, 0.028)],
        }
        self.vpp_cycle_place_pose = None
        self.vpp_cycle_kind = None
        self.vpp_stable_pose = None
        self.vpp_stable_ts = 0.0
        self.vpp_active = False
        self.vpp_state = "IDLE"
        self.vpp_state_ts = 0.0
        self.vpp_settle_ts = 0.0
        self.vpp_latched_box = None     # [bx, by, bz, yaw_box, j6_pick, h_box]
        self.vpp_grasp_locked = False
        self.vpp_confirmed_grasp_width = None
        self.vpp_confirmed_grasp_force = None
        self.vpp_latched_grasp = (55, 50.0, 1.2)  # Default 55mm for 6x6cm parts (width_mm, force_n, hold_delay_s)
        self.vpp_arc_start = None       # [x, y, z, j6]
        self.vpp_arc_end = None         # [x, y, z, j6]
        self.vpp_arc_peak_z = 0.36
        self.vpp_arc_s = 0.0
        self.vpp_arc_dur = 2.0
        self.vpp_arc_next_state = "DONE"

        # Emergency Stop threshold: 1 cm (0.01 m) per user request
        self.danger_margin = 0.01
        self.warn_margin = 0.15

        # Load initial program
        self.load_program_from_file("Pick_and_Place_NMPC")

        # Build UI
        self.init_ui()
        self.apply_dark_theme()

        # 50 Hz ROS and UI update timer
        self.ros_timer = QTimer(self)
        self.ros_timer.timeout.connect(self.system_tick)
        self.ros_timer.start(20) # 20 ms = 50 Hz

    def init_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(10, 8, 10, 8)
        main_layout.setSpacing(8)

        # 1. Top Header Bar
        header = self.create_header()
        main_layout.addWidget(header)

        # 2. Main Tab Widget (Compact 6 tabs — ordered per user request, zero scroll arrows)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setUsesScrollButtons(False)
        self.tabs.tabBar().setExpanding(True)

        self.tab_manager = self.create_manager_tab()
        self.tab_jog = self.tab_manager
        self.tab_dynamic = self.create_dynamic_tab()
        self.tab_vision_pick = self.create_vision_pick_tab()
        self.tab_builder = self.create_builder_tab()
        self.tab_nmpc = self.create_nmpc_tab()
        self.tab_launcher = self.create_launcher_tab()

        self.tabs.addTab(self.tab_manager, "1. Control & Safety")
        self.tabs.addTab(self.tab_dynamic, "2. Dynamic Lab")
        self.tabs.addTab(self.tab_vision_pick, "3. Vision Pick")
        self.tabs.addTab(self.tab_builder, "4. Task Builder")
        self.tabs.addTab(self.tab_nmpc, "5. NMPC Setup")
        self.tabs.addTab(self.tab_launcher, "6. Process")

        main_layout.addWidget(self.tabs)

    def create_dynamic_tab(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Card 1: Choose Simulation Problem (Rail Default vs Dynamic World)
        scen_box = QtWidgets.QGroupBox("1. Chọn Bài Toán Mô Phỏng (Tự động mở đúng thế giới Gazebo)")
        sb_lay = QtWidgets.QVBoxLayout(scen_box)
        sb_lay.setSpacing(8)

        scen_btn_row = QtWidgets.QHBoxLayout()
        scen_btn_row.setSpacing(10)
        self.btn_scen_static = QtWidgets.QPushButton("🛤️ BÀI TOÁN 1: THANH RAY TẠI HOME (pick_and_place_rail.sdf — Mặc định)")
        self.btn_scen_static.setCheckable(True)
        self.btn_scen_static.setChecked(True)
        self.btn_scen_static.clicked.connect(lambda: self.on_select_scenario("rail"))
        scen_btn_row.addWidget(self.btn_scen_static, stretch=1)

        self.btn_scen_dynamic = QtWidgets.QPushButton("⚡ BÀI TOÁN 2: VẬT ĐỘNG TRÒN / ZIGZAG (pick_and_place_dynamic.sdf)")
        self.btn_scen_dynamic.setCheckable(True)
        self.btn_scen_dynamic.clicked.connect(lambda: self.on_select_scenario("dynamic"))
        scen_btn_row.addWidget(self.btn_scen_dynamic, stretch=1)
        sb_lay.addLayout(scen_btn_row)

        info_row = QtWidgets.QHBoxLayout()
        self.lbl_scen_world_badge = QtWidgets.QLabel("World: pick_and_place_rail.sdf (Thanh ray tại HOME — Mặc định)")
        self.lbl_scen_world_badge.setStyleSheet("color: #38BDF8; font-family: monospace; font-size: 11px; font-weight: bold;")
        info_row.addWidget(self.lbl_scen_world_badge, stretch=1)

        self.chk_rail_slide = QtWidgets.QCheckBox("Cho Trụ Vàng trượt trên thanh ray 40cm")
        self.chk_rail_slide.setStyleSheet("color: #FDE047; font-weight: bold; font-size: 11px;")
        self.chk_rail_slide.toggled.connect(self.on_toggle_rail_slide)
        info_row.addWidget(self.chk_rail_slide)

        btn_reset_dyn_box = QtWidgets.QPushButton("🔄 Reset Phôi Băng Tải & Ô 2×2")
        btn_reset_dyn_box.setStyleSheet("background: #0F766E; color: white; font-weight: bold; padding: 6px 12px;")
        btn_reset_dyn_box.clicked.connect(self.reset_target_box_pose)
        info_row.addWidget(btn_reset_dyn_box)
        sb_lay.addLayout(info_row)
        layout.addWidget(scen_box)

        # Card 2: Dynamic Obstacle Type, Move/Pause Toggle, and Speed Control
        dyn_box = QtWidgets.QGroupBox("2. Điều Khiển Vật Cản Động (Loại Quỹ Đạo | Bật/Tắt Di Chuyển | Vận Tốc)")
        db_lay = QtWidgets.QVBoxLayout(dyn_box)
        db_lay.setSpacing(10)

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.setSpacing(10)
        self.lbl_dyn_kind = QtWidgets.QLabel("Chọn loại vật cản động:")
        self.lbl_dyn_kind.setStyleSheet("color: #FDBA74; font-weight: bold; font-size: 11px; min-width: 150px;")
        mode_row.addWidget(self.lbl_dyn_kind)

        self.btn_dyn_circle = QtWidgets.QPushButton("⭕ Vật Cản Hình Tròn (Quay Quanh Tâm)")
        self.btn_dyn_circle.setCheckable(True)
        self.btn_dyn_circle.setChecked(True)
        self.btn_dyn_circle.clicked.connect(lambda: self.on_select_dyn_mode("circle"))
        mode_row.addWidget(self.btn_dyn_circle, stretch=1)

        self.btn_dyn_zigzag = QtWidgets.QPushButton("⚡ Vật Cản Zigzag (Quỹ Đạo Khó Đoán)")
        self.btn_dyn_zigzag.setCheckable(True)
        self.btn_dyn_zigzag.clicked.connect(lambda: self.on_select_dyn_mode("zigzag"))
        mode_row.addWidget(self.btn_dyn_zigzag, stretch=1)
        db_lay.addLayout(mode_row)

        # Pause / Resume Moving Toggle + Velocity Controls
        ctrl_row = QtWidgets.QHBoxLayout()
        ctrl_row.setSpacing(10)
        self.btn_dyn_pause_toggle = QtWidgets.QPushButton("▶ ĐANG DI CHUYỂN (Bấm để Dừng Đứng Yên)")
        self.btn_dyn_pause_toggle.setCheckable(True)
        self.btn_dyn_pause_toggle.setChecked(False)
        self.btn_dyn_pause_toggle.clicked.connect(self.toggle_dyn_obstacle_motion)
        ctrl_row.addWidget(self.btn_dyn_pause_toggle, stretch=2)

        self.lbl_dyn_spd_txt = QtWidgets.QLabel("Vận tốc vật cản:")
        self.lbl_dyn_spd_txt.setStyleSheet("color: #CBD5E1; font-weight: bold; font-size: 11px;")
        ctrl_row.addWidget(self.lbl_dyn_spd_txt)

        self.sl_dyn_speed = QtWidgets.QSlider(Qt.Horizontal)
        self.sl_dyn_speed.setRange(2, 25)
        self.sl_dyn_speed.setValue(10)
        self.sl_dyn_speed.valueChanged.connect(self.on_dyn_speed_changed)
        ctrl_row.addWidget(self.sl_dyn_speed, stretch=2)

        self.lbl_dyn_speed = QtWidgets.QLabel("1.0x")
        self.lbl_dyn_speed.setStyleSheet("color: #FDBA74; font-weight: bold; font-family: monospace; font-size: 12px; min-width: 40px;")
        ctrl_row.addWidget(self.lbl_dyn_speed)

        for lbl_spd, val_spd in [("0.5x", 5), ("1.0x", 10), ("1.5x", 15), ("2.0x", 20)]:
            bsp = QtWidgets.QPushButton(lbl_spd)
            bsp.setStyleSheet("padding: 5px 9px; font-weight: bold; font-size: 10px;")
            bsp.clicked.connect(lambda _, v=val_spd: self.sl_dyn_speed.setValue(v))
            ctrl_row.addWidget(bsp)
        db_lay.addLayout(ctrl_row)
        layout.addWidget(dyn_box)

        # Card 3: Real-time Velocity & Pose Telemetry
        tele_box = QtWidgets.QGroupBox("3. Giám Sát Trạng Thái & Vận Tốc Vật Cản Động Thời Gian Thực")
        tb_lay = QtWidgets.QGridLayout(tele_box)
        tb_lay.setSpacing(8)

        self.lbl_dyn_live_state = _fix_label(
            QtWidgets.QLabel("Trạng thái: Đang ở Bài toán Vật Cản Tĩnh (Hãy bấm Bài toán 2 để bật vật cản động)"),
            fixed_h=18,
        )
        self.lbl_dyn_live_state.setStyleSheet("color: #94A3B8; font-weight: bold; font-size: 11px;")
        tb_lay.addWidget(self.lbl_dyn_live_state, 0, 0, 1, 2)

        self.lbl_dyn_live_pos = _fix_label(
            QtWidgets.QLabel("Vị trí XY: [---, ---] m | H = --- cm | R = --- cm"),
            fixed_h=18,
        )
        self.lbl_dyn_live_pos.setStyleSheet("color: #FDE68A; font-family: monospace; font-weight: bold; font-size: 12px;")
        tb_lay.addWidget(self.lbl_dyn_live_pos, 1, 0)

        self.lbl_dyn_live_vel = _fix_label(
            QtWidgets.QLabel("Vận tốc (Vx, Vy): [0.00, 0.00] m/s | Độ lớn |V|: 0.00 m/s"),
            fixed_h=18,
        )
        self.lbl_dyn_live_vel.setStyleSheet("color: #38BDF8; font-family: monospace; font-weight: bold; font-size: 12px;")
        tb_lay.addWidget(self.lbl_dyn_live_vel, 1, 1)

        note_lbl = QtWidgets.QLabel(
            "💡 Mẹo: Khi chuyển sang Bài toán 2, Gazebo sẽ mở thế giới động có đủ Hộp Vàng + Khay A/B. "
            "Bạn có thể bật/tắt di chuyển hoặc chỉnh vận tốc ngay trong lúc robot đang chạy gắp thả ở Tab 6 hoặc Tab 1!"
        )
        note_lbl.setWordWrap(True)
        note_lbl.setStyleSheet("color: #94A3B8; font-size: 11px;")
        tb_lay.addWidget(note_lbl, 2, 0, 1, 2)
        layout.addWidget(tele_box)

        layout.addStretch()
        self._refresh_scenario_styles()
        return w

    def create_header(self):
        w = QtWidgets.QFrame()
        w.setObjectName("headerFrame")
        layout = QtWidgets.QHBoxLayout(w)
        layout.setContentsMargins(12, 6, 12, 6)

        # Branding
        brand_layout = QtWidgets.QHBoxLayout()
        dot = QtWidgets.QLabel("●")
        dot.setStyleSheet("color: #10B981; font-size: 16px;")
        title = QtWidgets.QLabel("DOOSAN ROBOTICS")
        title.setStyleSheet("font-weight: 900; font-size: 15px; color: #38BDF8; letter-spacing: 1px;")
        sub = QtWidgets.QLabel("M0609 Cobot | DART-Platform v3.2")
        sub.setStyleSheet("color: #94A3B8; font-size: 11px;")
        brand_layout.addWidget(dot)
        brand_layout.addWidget(title)
        brand_layout.addWidget(sub)
        layout.addLayout(brand_layout)

        layout.addStretch()

        # Status pills
        servo_pill = QtWidgets.QLabel("SERVO ON")
        servo_pill.setStyleSheet("background: #064E3B; color: #34D399; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 10px;")
        layout.addWidget(servo_pill)

        self.safety_badge = QtWidgets.QLabel("SAFETY OK")
        self.safety_badge.setStyleSheet("background: #064E3B; color: #34D399; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 10px;")
        layout.addWidget(self.safety_badge)

        # Speed Override
        speed_box = QtWidgets.QHBoxLayout()
        speed_lbl = QtWidgets.QLabel("Speed:")
        speed_lbl.setStyleSheet("color: #94A3B8; font-size: 11px;")
        self.speed_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.speed_slider.setRange(10, 100)
        self.speed_slider.setValue(80)
        self.speed_slider.setFixedWidth(90)
        self.speed_val_lbl = QtWidgets.QLabel("80%")
        self.speed_val_lbl.setStyleSheet("color: #38BDF8; font-weight: bold; font-size: 11px; min-width: 32px;")
        self.speed_slider.valueChanged.connect(self._on_global_speed_changed)
        speed_box.addWidget(speed_lbl)
        speed_box.addWidget(self.speed_slider)
        speed_box.addWidget(self.speed_val_lbl)
        layout.addLayout(speed_box)

        # E-STOP Button
        self.btn_estop = QtWidgets.QPushButton("E-STOP")
        self.btn_estop.setStyleSheet("""
            QPushButton {
                background: #DC2626; color: white; font-weight: 900; font-size: 12px;
                padding: 6px 14px; border-radius: 6px; border: 1px solid #F87171;
            }
            QPushButton:hover { background: #B91C1C; }
        """)
        self.btn_estop.clicked.connect(self.trigger_estop)
        layout.addWidget(self.btn_estop)

        # EXIT SYSTEM BUTTON (Kills all processes cleanly)
        btn_exit = QtWidgets.QPushButton("🚪 EXIT SYSTEM")
        btn_exit.setStyleSheet("""
            QPushButton {
                background: #1E293B; color: #FCA5A5; font-weight: bold; font-size: 11px;
                padding: 6px 12px; border-radius: 6px; border: 1px solid #7F1D1D;
            }
            QPushButton:hover { background: #7F1D1D; color: white; }
        """)
        btn_exit.clicked.connect(self.confirm_and_exit_system)
        layout.addWidget(btn_exit)

        return w

    # ========================================================
    # TAB 1: UNIFIED CONTROL, SAFETY, JOG, GRIPPER & VISION D435
    # ========================================================
    def create_manager_tab(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # 1. Master Program Execution Strip
        exec_frame = QtWidgets.QFrame()
        exec_frame.setObjectName("cardFrame")
        ef_layout = QtWidgets.QHBoxLayout(exec_frame)
        ef_layout.setContentsMargins(8, 5, 8, 5)
        ef_layout.setSpacing(5)

        self.btn_start = QtWidgets.QPushButton("▶ START")
        self.btn_start.setStyleSheet(
            "QPushButton { background: #059669; color: white; font-weight: bold; font-size: 11px; padding: 5px 11px; border-radius: 5px; border: 1px solid #34D399; }"
            "QPushButton:hover { background: #047857; }"
        )
        self.btn_start.clicked.connect(self.start_program)

        self.btn_pause = QtWidgets.QPushButton("⏸ PAUSE")
        self.btn_pause.setStyleSheet("padding: 5px 9px; font-weight: bold; background: #D97706; color: white;")
        self.btn_pause.clicked.connect(self.pause_program)

        self.btn_mode = QtWidgets.QPushButton("CHẾ ĐỘ: AUTO")
        self.btn_mode.setStyleSheet("padding: 5px 9px; font-weight: bold; background: #2563EB; color: white;")
        self.btn_mode.clicked.connect(self.toggle_auto_manual_mode)

        self.btn_check_go = QtWidgets.QPushButton("⏭ GO")
        self.btn_check_go.setStyleSheet("padding: 5px 8px; font-weight: bold; background: #1E40AF; color: white;")
        self.btn_check_go.clicked.connect(self.check_go_next_step)

        self.btn_check_back = QtWidgets.QPushButton("⏮ BACK")
        self.btn_check_back.setStyleSheet("padding: 5px 8px; font-weight: bold; background: #475569; color: white;")
        self.btn_check_back.clicked.connect(self.check_back_prev_step)

        self.btn_reset = QtWidgets.QPushButton("⏹ RESET")
        self.btn_reset.setStyleSheet("padding: 5px 9px; font-weight: bold; background: #DC2626; color: white;")
        self.btn_reset.clicked.connect(self.reset_program)

        self.btn_reset_box = QtWidgets.QPushButton("📦 RESET PHÔI")
        self.btn_reset_box.setStyleSheet("padding: 5px 10px; font-weight: bold; background: #0D9488; color: white;")
        self.btn_reset_box.setToolTip("Đặt lại Hộp Cam, Trụ Xanh trên băng tải & reset ô 2×2")
        self.btn_reset_box.clicked.connect(self.reset_target_box_pose)

        for b in (self.btn_start, self.btn_pause, self.btn_mode, self.btn_check_go, self.btn_check_back, self.btn_reset, self.btn_reset_box):
            ef_layout.addWidget(b)
        ef_layout.addStretch()

        self.lbl_prog_name = QtWidgets.QLabel(f"Prog: {self.current_program_name}")
        self.lbl_prog_name.setStyleSheet("color: #38BDF8; font-weight: bold; font-family: monospace; font-size: 11px;")
        self.lbl_prog_step = QtWidgets.QLabel("Step: Sẵn sàng")
        self.lbl_prog_step.setStyleSheet("color: #34D399; font-weight: bold; font-family: monospace; font-size: 11px;")
        self.lbl_timer = QtWidgets.QLabel("00:00.0s")
        self.lbl_timer.setStyleSheet("color: #FFFFFF; font-weight: bold; font-family: monospace; font-size: 11px;")

        ef_layout.addWidget(self.lbl_prog_name)
        ef_layout.addSpacing(8)
        ef_layout.addWidget(self.lbl_prog_step)
        ef_layout.addSpacing(8)
        ef_layout.addWidget(self.lbl_timer)
        layout.addWidget(exec_frame)

        # 2. Main Unified 3-Column Body: [Radar + 3D Safety] | [Jog + Gripper / 6-DOF Sub-Tabs] | [D435 Vision Cockpit]
        mid_layout = QtWidgets.QHBoxLayout()
        mid_layout.setSpacing(8)

        # --- COL 1: 2D Radar + 3D Clearance + TCP Telemetry ---
        left_col = QtWidgets.QWidget()
        lc_lay = QtWidgets.QVBoxLayout(left_col)
        lc_lay.setContentsMargins(0, 0, 0, 0)
        lc_lay.setSpacing(6)

        radar_group = QtWidgets.QGroupBox("2D Workspace Radar (Click Jog)")
        radar_layout = QtWidgets.QVBoxLayout(radar_group)
        radar_layout.setContentsMargins(6, 8, 6, 6)
        self.radar = ActiveWorkspaceRadar()
        self.radar.setMinimumHeight(210)
        self.radar.target_clicked.connect(self.on_radar_click)
        radar_layout.addWidget(self.radar)
        lc_lay.addWidget(radar_group, stretch=1)

        safety_group = QtWidgets.QGroupBox("Giám sát Khoảng cách 3D & Telemetry")
        sg_layout = QtWidgets.QVBoxLayout(safety_group)
        sg_layout.setContentsMargins(8, 8, 8, 6)
        sg_layout.setSpacing(4)

        dist_row = QtWidgets.QHBoxLayout()
        dist_lbl = QtWidgets.QLabel("Khoảng cách 3D tới cản:")
        self.dist_val_lbl = QtWidgets.QLabel("18.4 cm")
        self.dist_val_lbl.setStyleSheet("font-size: 15px; font-weight: 900; color: #34D399; font-family: monospace;")
        dist_row.addWidget(dist_lbl)
        dist_row.addStretch()
        dist_row.addWidget(self.dist_val_lbl)
        sg_layout.addLayout(dist_row)

        self.dist_bar = QtWidgets.QProgressBar()
        self.dist_bar.setRange(0, 30)
        self.dist_bar.setValue(18)
        self.dist_bar.setTextVisible(False)
        self.dist_bar.setFixedHeight(9)
        self.dist_bar.setStyleSheet("QProgressBar { background: #1E293B; border-radius: 4px; } QProgressBar::chunk { background: #10B981; border-radius: 4px; }")
        sg_layout.addWidget(self.dist_bar)

        breakdown_layout = QtWidgets.QHBoxLayout()
        self.lbl_delta_z = QtWidgets.QLabel("Δz: +10.5 cm")
        self.lbl_delta_z.setStyleSheet("font-family: monospace; font-size: 10px; color: #38BDF8;")
        self.lbl_delta_z.setMinimumWidth(130)
        _fix_label(self.lbl_delta_z, fixed_h=16)
        self.lbl_delta_r = QtWidgets.QLabel("Δr: +15.2 cm")
        self.lbl_delta_r.setStyleSheet("font-family: monospace; font-size: 10px; color: #94A3B8;")
        self.lbl_delta_r.setMinimumWidth(130)
        _fix_label(self.lbl_delta_r, fixed_h=16)
        breakdown_layout.addWidget(self.lbl_delta_z)
        breakdown_layout.addWidget(self.lbl_delta_r)
        sg_layout.addLayout(breakdown_layout)

        self.safety_desc = _fix_label(QtWidgets.QLabel("✓ AN TOÀN (AUTO E-STOP: 1.0 cm)"), fixed_h=18)
        self.safety_desc.setStyleSheet("color: #34D399; font-weight: bold; font-size: 10px;")
        sg_layout.addWidget(self.safety_desc)

        self.lbl_tele_xyz = QtWidgets.QLabel("TCP XYZ: [0.368, 0.006, 0.236] m")
        self.lbl_tele_xyz.setStyleSheet("font-family: monospace; font-size: 10px; color: #FFFFFF;")
        self.lbl_tele_pitch = QtWidgets.QLabel("Tool Pitch: -90.0° (Thẳng đứng)")
        self.lbl_tele_pitch.setStyleSheet("font-family: monospace; font-size: 10px; color: #34D399;")
        self.lbl_tele_grip = QtWidgets.QLabel("Robotiq 2F-85: 55 mm")
        self.lbl_tele_grip.setStyleSheet("font-family: monospace; font-size: 10px; color: #38BDF8;")
        self.lbl_tele_nmpc = QtWidgets.QLabel("NMPC RTI Latency: 1.18 ms (OK)")
        self.lbl_tele_nmpc.setStyleSheet("font-family: monospace; font-size: 10px; color: #A78BFA;")
        for lbl_t in (self.lbl_tele_xyz, self.lbl_tele_pitch, self.lbl_tele_grip, self.lbl_tele_nmpc):
            _fix_label(lbl_t, fixed_h=16)
            sg_layout.addWidget(lbl_t)
        lc_lay.addWidget(safety_group)
        mid_layout.addWidget(left_col, stretch=4)

        # --- COL 2: Compact Inner Tab [Jog + Gripper | 6-DOF Joints + XYZ] ---
        mid_sub = QtWidgets.QTabWidget()
        mid_sub.setDocumentMode(True)

        # Sub-tab A: Dobot Jog + Gripper Width & Force
        sub_jog_w = QtWidgets.QWidget()
        sj_lay = QtWidgets.QVBoxLayout(sub_jog_w)
        sj_lay.setContentsMargins(4, 6, 4, 4)
        sj_lay.setSpacing(6)

        self.jog_panel_tab3 = self._create_synced_jog_panel(compact=True)
        sj_lay.addWidget(self.jog_panel_tab3)

        grip_box = QtWidgets.QGroupBox("Robotiq 2F-85 & Lực Kẹp")
        g_layout = QtWidgets.QVBoxLayout(grip_box)
        g_layout.setContentsMargins(6, 8, 6, 5)
        g_layout.setSpacing(3)

        self.gripper_viz = GripperVisualWidget()
        self.gripper_viz.setFixedHeight(78)
        g_layout.addWidget(self.gripper_viz)

        self.lbl_grip_status = QtWidgets.QLabel("🟢 ĐANG MỞ/KẸP (55 mm)")
        self.lbl_grip_status.setStyleSheet("font-weight: bold; color: #FBBF24; font-size: 10px;")
        g_layout.addWidget(self.lbl_grip_status)

        self.grip_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.grip_slider.setRange(0, 85)
        self.grip_slider.setValue(55)
        self.grip_slider.valueChanged.connect(self.on_gripper_slider)
        g_layout.addWidget(self.grip_slider)

        p_row = QtWidgets.QHBoxLayout()
        p_row.setSpacing(3)
        for p_txt, p_val, p_bg in [("Mở 85", 85, None), ("Phôi 55", 55, "#B45309"), ("Chặt 48", 48, None), ("Đóng 0", 0, None)]:
            bp = QtWidgets.QPushButton(p_txt)
            bp.setStyleSheet(f"background: {p_bg}; color: white; padding: 3px 5px; font-size: 10px;" if p_bg else "padding: 3px 5px; font-size: 10px;")
            bp.clicked.connect(lambda _, v=p_val: self.grip_slider.setValue(v))
            p_row.addWidget(bp)
        g_layout.addLayout(p_row)

        self.lbl_force_val = QtWidgets.QLabel("💪 Lực kẹp: 50 N (Chống bẹp phôi)")
        self.lbl_force_val.setStyleSheet("color: #38BDF8; font-size: 10px; font-family: monospace; font-weight: bold;")
        g_layout.addWidget(self.lbl_force_val)

        force_row = QtWidgets.QHBoxLayout()
        force_row.setSpacing(3)
        self.force_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.force_slider.setRange(10, 120)
        self.force_slider.setValue(50)
        self.force_slider.valueChanged.connect(self.on_force_slider)
        force_row.addWidget(self.force_slider, stretch=1)
        for ft, fv in [("25N", 25), ("50N", 50), ("90N", 90)]:
            bf = QtWidgets.QPushButton(ft)
            bf.setStyleSheet("padding: 2px 6px; font-size: 10px;")
            bf.clicked.connect(lambda _, v=fv: self.force_slider.setValue(v))
            force_row.addWidget(bf)
        g_layout.addLayout(force_row)
        sj_lay.addWidget(grip_box)
        sj_lay.addStretch()
        mid_sub.addTab(sub_jog_w, "🕹️ Jog & Gripper")

        # Sub-tab B: 6-DOF Joints + Target XYZ Input
        sub_jnt_w = QtWidgets.QWidget()
        sjn_lay = QtWidgets.QVBoxLayout(sub_jnt_w)
        sjn_lay.setContentsMargins(4, 6, 4, 4)
        sjn_lay.setSpacing(6)

        target_group = QtWidgets.QGroupBox("🎯 Nhập Tọa Độ Mục Tiêu (m)")
        tg_layout = QtWidgets.QVBoxLayout(target_group)
        tg_layout.setContentsMargins(6, 8, 6, 6)
        tg_layout.setSpacing(4)
        inp_grid = QtWidgets.QGridLayout()
        inp_grid.setSpacing(4)
        inp_grid.addWidget(QtWidgets.QLabel("X:"), 0, 0)
        self.inp_target_x = QtWidgets.QDoubleSpinBox()
        self.inp_target_x.setRange(-1.0, 1.0); self.inp_target_x.setValue(self.tcp_pos[0]); self.inp_target_x.setSingleStep(0.01); self.inp_target_x.setDecimals(3)
        inp_grid.addWidget(self.inp_target_x, 0, 1)
        inp_grid.addWidget(QtWidgets.QLabel("Y:"), 0, 2)
        self.inp_target_y = QtWidgets.QDoubleSpinBox()
        self.inp_target_y.setRange(-1.0, 1.0); self.inp_target_y.setValue(self.tcp_pos[1]); self.inp_target_y.setSingleStep(0.01); self.inp_target_y.setDecimals(3)
        inp_grid.addWidget(self.inp_target_y, 0, 3)
        inp_grid.addWidget(QtWidgets.QLabel("Z:"), 1, 0)
        self.inp_target_z = QtWidgets.QDoubleSpinBox()
        self.inp_target_z.setRange(0.0, 1.0); self.inp_target_z.setValue(self.tcp_pos[2]); self.inp_target_z.setSingleStep(0.01); self.inp_target_z.setDecimals(3)
        inp_grid.addWidget(self.inp_target_z, 1, 1)
        btn_get_curr = QtWidgets.QPushButton("Lấy TCP")
        btn_get_curr.clicked.connect(self.fill_target_with_current)
        inp_grid.addWidget(btn_get_curr, 1, 2, 1, 2)
        tg_layout.addLayout(inp_grid)

        act_row = QtWidgets.QHBoxLayout()
        btn_move_target = QtWidgets.QPushButton("🚀 ĐI TỚI XYZ")
        btn_move_target.setStyleSheet("background: #0284C7; color: white; font-weight: bold; padding: 5px;")
        btn_move_target.clicked.connect(self.move_to_input_target)
        btn_chot = QtWidgets.QPushButton("+ LƯU TREE →")
        btn_chot.setStyleSheet("background: #059669; color: white; font-weight: bold; padding: 5px;")
        btn_chot.clicked.connect(self.record_and_switch_builder)
        act_row.addWidget(btn_move_target)
        act_row.addWidget(btn_chot)
        tg_layout.addLayout(act_row)
        sjn_lay.addWidget(target_group)

        joint_box = QtWidgets.QGroupBox("6-DOF Joint Jogging & Poses")
        j_layout = QtWidgets.QVBoxLayout(joint_box)
        j_layout.setContentsMargins(6, 8, 6, 6)
        j_layout.setSpacing(4)
        pose_row = QtWidgets.QHBoxLayout()
        for pt, pq in [("🏠 HOME", [0.0, 0.0, 1.5708, 0.0, 1.5708, 0.0]), ("🎯 READY", [0.0, 0.25, 1.30, 0.0, 1.57, 0.0]), ("🔄 ZERO", [0.0]*6)]:
            bp = QtWidgets.QPushButton(pt)
            bp.setStyleSheet("padding: 4px 6px; font-size: 10px; font-weight: bold;")
            bp.clicked.connect(lambda _, qv=pq: self.send_joint_pose(qv))
            pose_row.addWidget(bp)
        j_layout.addLayout(pose_row)

        self.joint_sliders, self.joint_labels = [], []
        for i in range(6):
            j_row = QtWidgets.QHBoxLayout()
            j_row.setSpacing(3)
            lbl = QtWidgets.QLabel(f"J{i+1}: 0.0°")
            lbl.setStyleSheet("font-family: monospace; font-size: 10px; min-width: 54px;")
            slider = QtWidgets.QSlider(Qt.Horizontal)
            slider.setRange(-180, 180); slider.setValue(0)
            slider.valueChanged.connect(lambda v, idx=i: self.on_joint_slider(idx, v))
            btn_minus = ContinuousJogButton("-")
            btn_minus.setFixedSize(20, 20); btn_minus.setStyleSheet("padding: 0px; font-weight: bold;")
            btn_minus.jog_signal.connect(lambda idx=i: self.jog_joint_fine(idx, -1.0))
            btn_plus = ContinuousJogButton("+")
            btn_plus.setFixedSize(20, 20); btn_plus.setStyleSheet("padding: 0px; font-weight: bold;")
            btn_plus.jog_signal.connect(lambda idx=i: self.jog_joint_fine(idx, 1.0))
            self.joint_labels.append(lbl); self.joint_sliders.append(slider)
            j_row.addWidget(lbl); j_row.addWidget(btn_minus); j_row.addWidget(slider); j_row.addWidget(btn_plus)
            j_layout.addLayout(j_row)
        sjn_lay.addWidget(joint_box, stretch=1)
        mid_sub.addTab(sub_jnt_w, "🎛️ 6-DOF & XYZ")

        mid_layout.addWidget(mid_sub, stretch=3)

        # --- COL 3: Compact RealSense D435 Vision Cockpit ---
        col3_box = QtWidgets.QGroupBox("📷 Camera D435 Cockpit")
        c3_layout = QtWidgets.QVBoxLayout(col3_box)
        c3_layout.setContentsMargins(6, 8, 6, 6)
        c3_layout.setSpacing(5)

        top_strip = QtWidgets.QHBoxLayout()
        top_strip.setSpacing(5)
        self.lbl_cam_stream_badge = _fix_label(QtWidgets.QLabel("⚪ Chờ D435..."), fixed_h=16)
        self.lbl_cam_stream_badge.setStyleSheet("color: #94A3B8; font-weight: bold; font-family: monospace; font-size: 10px;")
        top_strip.addWidget(self.lbl_cam_stream_badge, stretch=1)

        btn_vis_start = QtWidgets.QPushButton("▶ Vision")
        btn_vis_start.setStyleSheet("background: #059669; color: white; font-weight: bold; padding: 3px 8px; font-size: 10px;")
        btn_vis_start.clicked.connect(self.start_vision_process)
        btn_vis_stop = QtWidgets.QPushButton("⏹ Tắt")
        btn_vis_stop.setStyleSheet("background: #7F1D1D; color: white; padding: 3px 8px; font-size: 10px;")
        btn_vis_stop.clicked.connect(self.stop_vision_process)
        top_strip.addWidget(btn_vis_start)
        top_strip.addWidget(btn_vis_stop)
        c3_layout.addLayout(top_strip)

        self.lbl_vision_video = QtWidgets.QLabel("Chưa có khung hình D435.\nHãy bật Gazebo (Tab 6) và bấm '▶ Vision'.")
        self.lbl_vision_video.setAlignment(Qt.AlignCenter)
        self.lbl_vision_video.setMinimumSize(300, 215)
        self.lbl_vision_video.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.lbl_vision_video.setStyleSheet(
            "background-color: #05080F; color: #64748B; border: 1px solid #334155; "
            "border-radius: 6px; font-size: 11px; font-weight: bold;"
        )
        c3_layout.addWidget(self.lbl_vision_video, stretch=1)

        obj_grid = QtWidgets.QGridLayout()
        obj_grid.setSpacing(4)

        def _make_obj_card(title_text, title_color, line1, line1_color, line2, line2_color):
            card = QtWidgets.QFrame()
            card.setObjectName("cardFrame")
            lay = QtWidgets.QVBoxLayout(card)
            lay.setContentsMargins(6, 4, 6, 4)
            lay.setSpacing(1)
            t_lbl = _fix_label(QtWidgets.QLabel(title_text), fixed_h=16)
            t_lbl.setStyleSheet(f"font-weight: bold; color: {title_color}; font-size: 10px;")
            l1 = _fix_label(QtWidgets.QLabel(line1), fixed_h=16)
            l1.setStyleSheet(f"font-family: monospace; font-size: 10px; font-weight: bold; color: {line1_color};")
            l2 = _fix_label(QtWidgets.QLabel(line2), fixed_h=16)
            l2.setStyleSheet(f"font-family: monospace; font-size: 10px; color: {line2_color};")
            lay.addWidget(t_lbl); lay.addWidget(l1); lay.addWidget(l2)
            return card, lay, l1, l2

        box_card, bc_layout, self.lbl_vis_box_xyz, self.lbl_vis_box_yaw = _make_obj_card(
            "📦 Phôi Băng Tải (Active)", "#F97316",
            "XYZ: ---, ---, --- m", "#FDBA74", "Yaw: ---° | --- cm", "#38BDF8"
        )
        self.lbl_vis_box_status = _fix_label(QtWidgets.QLabel("Chờ nhận diện"), fixed_h=16)
        self.lbl_vis_box_status.setStyleSheet("font-size: 10px; color: #94A3B8;")
        bc_layout.addWidget(self.lbl_vis_box_status)
        obj_grid.addWidget(box_card, 0, 0)

        obs_card, oc_layout, self.lbl_vis_obs_xyz, self.lbl_vis_obs_dim = _make_obj_card(
            "🟡 Cản Vàng (Ø10×26cm)", "#FACC15",
            "XY: ---, --- m | R: ---", "#FEF08A", "W×D×H: --- cm", "#CBD5E1"
        )
        self.chk_vision_sync_obs = QtWidgets.QCheckBox("Đồng bộ Cản → NMPC")
        self.chk_vision_sync_obs.setStyleSheet("color: #34D399; font-weight: bold; font-size: 10px;")
        self.chk_vision_sync_obs.toggled.connect(self.on_toggle_vision_obs_sync)
        oc_layout.addWidget(self.chk_vision_sync_obs)
        obj_grid.addWidget(obs_card, 0, 1)

        slot3_card, _, self.lbl_vis_slot3_xyz, self.lbl_vis_slot3_info = _make_obj_card(
            "🟧 Vuông Cam (6×6×6)", "#FB923C",
            "XYZ: ---, ---, ---", "#FED7AA", "Chưa thấy", "#94A3B8"
        )
        obj_grid.addWidget(slot3_card, 1, 0)

        slot4_card, _, self.lbl_vis_slot4_xyz, self.lbl_vis_slot4_info = _make_obj_card(
            "🟢 Trụ Xanh (Ø6×6)", "#4ADE80",
            "XYZ: ---, ---, ---", "#BBF7D0", "Chưa thấy", "#94A3B8"
        )
        obj_grid.addWidget(slot4_card, 1, 1)

        c3_layout.addLayout(obj_grid)
        mid_layout.addWidget(col3_box, stretch=4)
        layout.addLayout(mid_layout, stretch=1)

        # 3. Bottom Active Program Step Tracker
        tracker_group = QtWidgets.QGroupBox("Chuỗi lệnh đang thực thi")
        self.tracker_layout = QtWidgets.QHBoxLayout(tracker_group)
        self.tracker_layout.setContentsMargins(6, 6, 6, 4)
        layout.addWidget(tracker_group)

        self.update_tracker_display()
        return w

    # ========================================================
    # TAB 2: DART TASK BUILDER (SPLIT VIEW & CARD STYLING)
    # ========================================================
    def create_builder_tab(self):
        w = QtWidgets.QWidget()
        main_h = QtWidgets.QHBoxLayout(w)
        main_h.setSpacing(12)

        # LEFT PANE: Program Tree (7 cols)
        left_box = QtWidgets.QGroupBox("Program Tree (Chuỗi câu lệnh - Card View)")
        left_layout = QtWidgets.QVBoxLayout(left_box)

        # Program Management Bar
        pm_bar = QtWidgets.QHBoxLayout()
        name_lbl = QtWidgets.QLabel("Tên Program:")
        self.edit_prog_name = QtWidgets.QLineEdit(self.current_program_name)
        self.edit_prog_name.setStyleSheet("font-family: monospace; font-weight: bold; color: #38BDF8;")

        btn_save = QtWidgets.QPushButton("💾 Lưu Program")
        btn_save.setStyleSheet("background: #0284C7; color: white; font-weight: bold;")
        btn_save.clicked.connect(self.save_program_dialog)

        btn_load = QtWidgets.QPushButton("📂 Mở Program Đã Lưu")
        btn_load.setStyleSheet("background: #4F46E5; color: white; font-weight: bold;")
        btn_load.clicked.connect(self.open_load_dialog)

        pm_bar.addWidget(name_lbl)
        pm_bar.addWidget(self.edit_prog_name)
        pm_bar.addWidget(btn_save)
        pm_bar.addWidget(btn_load)
        left_layout.addLayout(pm_bar)

        # Scroll area containing beautiful card widgets
        self.tree_scroll = QtWidgets.QScrollArea()
        self.tree_scroll.setWidgetResizable(True)
        self.tree_scroll.setStyleSheet("background: #0A101C; border: 1px solid #1E293B; border-radius: 8px;")
        
        self.tree_container = QtWidgets.QWidget()
        self.tree_container_layout = QtWidgets.QVBoxLayout(self.tree_container)
        self.tree_container_layout.setContentsMargins(6, 6, 6, 6)
        self.tree_container_layout.setSpacing(6)
        self.tree_container_layout.addStretch()
        self.tree_scroll.setWidget(self.tree_container)

        left_layout.addWidget(self.tree_scroll)

        # Bottom Tree Operations
        tree_ops = QtWidgets.QHBoxLayout()
        btn_new = QtWidgets.QPushButton("Mới (+)")
        btn_new.clicked.connect(self.new_program)

        btn_clear = QtWidgets.QPushButton("Clear All")
        btn_clear.setStyleSheet("background: #7F1D1D; color: #FCA5A5;")
        btn_clear.clicked.connect(self.clear_program_tree)

        btn_apply = QtWidgets.QPushButton("✓ Nạp sang Tab 1")
        btn_apply.setStyleSheet("background: #059669; color: white; font-weight: bold; padding: 6px 16px;")
        btn_apply.clicked.connect(self.apply_to_manager)

        tree_ops.addWidget(btn_new)
        tree_ops.addWidget(btn_clear)
        tree_ops.addStretch()
        tree_ops.addWidget(btn_apply)
        left_layout.addLayout(tree_ops)

        main_h.addWidget(left_box, stretch=7)

        # RIGHT PANE: Command Palette & Waypoint Teaching (5 cols)
        right_box = QtWidgets.QGroupBox("Command Palette (Kho khối lệnh)")
        right_layout = QtWidgets.QVBoxLayout(right_box)
        right_layout.setSpacing(10)

        # Direct Teaching Banner
        teach_box = QtWidgets.QFrame()
        teach_box.setObjectName("cardFrame")
        tb_layout = QtWidgets.QVBoxLayout(teach_box)
        
        teach_title = QtWidgets.QLabel("📌 Dạy điểm trực tiếp từ Jogging:")
        teach_title.setStyleSheet("font-weight: bold; color: #38BDF8; font-size: 12px;")
        self.lbl_palette_tcp = _fix_label(
            QtWidgets.QLabel(f"Tọa độ hiện tại: [{self.tcp_pos[0]:.3f}, {self.tcp_pos[1]:.3f}, {self.tcp_pos[2]:.3f}] m"),
            fixed_h=18,
        )
        self.lbl_palette_tcp.setStyleSheet("font-family: monospace; font-size: 11px; color: #34D399;")
        
        btn_record_curr = QtWidgets.QPushButton("+ LƯU TỌA ĐỘ HIỆN TẠI VÀO TREE")
        btn_record_curr.setStyleSheet("background: #0284C7; color: white; font-weight: bold; padding: 8px;")
        btn_record_curr.clicked.connect(self.record_current_tcp_to_tree)

        btn_record_vision = QtWidgets.QPushButton("📷 NẠP TỌA ĐỘ HỘP TỪ VISION D435")
        btn_record_vision.setStyleSheet("background: #7C3AED; color: white; font-weight: bold; padding: 8px;")
        btn_record_vision.clicked.connect(self.record_vision_box_to_tree)

        tb_layout.addWidget(teach_title)
        tb_layout.addWidget(self.lbl_palette_tcp)
        tb_layout.addWidget(btn_record_curr)
        tb_layout.addWidget(btn_record_vision)
        right_layout.addWidget(teach_box)

        # Standard Command Buttons
        btn_nmpc = QtWidgets.QPushButton("Move theo NMPC (Tự giải quỹ đạo né cản)")
        btn_nmpc.setStyleSheet("text-align: left; padding: 10px; font-weight: bold; color: #38BDF8; border: 1px solid #0284C7;")
        btn_nmpc.clicked.connect(lambda: self.add_command_to_tree('MoveNMPC'))
        right_layout.addWidget(btn_nmpc)

        btn_orbit = QtWidgets.QPushButton("Quay hình tròn xung quanh 1 điểm (Orbit)")
        btn_orbit.setStyleSheet("text-align: left; padding: 10px; font-weight: bold; color: #C084FC; border: 1px solid #7C3AED;")
        btn_orbit.clicked.connect(lambda: self.add_command_to_tree('RotateCircle'))
        right_layout.addWidget(btn_orbit)

        btn_movej = QtWidgets.QPushButton("Move Joint (MoveJ - Về Home an toàn)")
        btn_movej.setStyleSheet("text-align: left; padding: 10px; font-weight: bold; color: #34D399; border: 1px solid #059669;")
        btn_movej.clicked.connect(lambda: self.add_command_to_tree('MoveJ'))
        right_layout.addWidget(btn_movej)

        btn_grip = QtWidgets.QPushButton("Robotiq Gripper Action (Kẹp / Nhả)")
        btn_grip.setStyleSheet("text-align: left; padding: 10px; font-weight: bold; color: #FBBF24; border: 1px solid #D97706;")
        btn_grip.clicked.connect(lambda: self.add_command_to_tree('Gripper'))
        right_layout.addWidget(btn_grip)

        btn_wait = QtWidgets.QPushButton("Delay / Wait Time (Tạm dừng)")
        btn_wait.setStyleSheet("text-align: left; padding: 10px; font-weight: bold; color: #94A3B8; border: 1px solid #475569;")
        btn_wait.clicked.connect(lambda: self.add_command_to_tree('Wait'))
        right_layout.addWidget(btn_wait)

        right_layout.addStretch()

        # Preset Coordinates Drawer
        coord_box = QtWidgets.QGroupBox("Tọa độ Mẫu Chuẩn")
        cb_layout = QtWidgets.QVBoxLayout(coord_box)
        cb_layout.addWidget(QtWidgets.QLabel("• Khay Gắp (A): [0.45, -0.28, 0.05] m"))
        cb_layout.addWidget(QtWidgets.QLabel("• Đỉnh Cột Cản: [0.45, 0.00, 0.28] m"))
        cb_layout.addWidget(QtWidgets.QLabel("• Khay Thả (B): [0.45, 0.28, 0.055] m"))
        right_layout.addWidget(coord_box)

        main_h.addWidget(right_box, stretch=5)

        self.refresh_tree_cards()
        return w

    # ========================================================
    # TAB 3: VISION PICK & PLACE (AUTO 2x2 CONTINUOUS vs MANUAL 1-SHOT)
    # ========================================================
    def create_vision_pick_tab(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ----------------------------------------------------
        # LEFT COLUMN (stretch=5): Live Camera D435 Feed + Quick Controls
        # ----------------------------------------------------
        left_box = QtWidgets.QGroupBox("📷 Camera D435 Live View (Giữ nguyên khung hình & Chọn Vật)")
        l_layout = QtWidgets.QVBoxLayout(left_box)
        l_layout.setContentsMargins(8, 10, 8, 8)
        l_layout.setSpacing(6)

        top_strip = QtWidgets.QHBoxLayout()
        top_strip.setSpacing(6)
        self.lbl_cam_stream_badge_tab6 = _fix_label(QtWidgets.QLabel("⚪ Đang chờ tín hiệu D435..."), fixed_h=18)
        self.lbl_cam_stream_badge_tab6.setStyleSheet("color: #94A3B8; font-weight: bold; font-family: monospace; font-size: 10px;")
        top_strip.addWidget(self.lbl_cam_stream_badge_tab6, stretch=1)

        btn_v6_start = QtWidgets.QPushButton("▶ Bật Vision")
        btn_v6_start.setStyleSheet("background: #059669; color: white; font-weight: bold; padding: 4px 10px; font-size: 10px;")
        btn_v6_start.clicked.connect(self.start_vision_process)
        top_strip.addWidget(btn_v6_start)

        btn_v6_stop = QtWidgets.QPushButton("⏹ Tắt")
        btn_v6_stop.setStyleSheet("background: #7F1D1D; color: white; padding: 4px 10px; font-size: 10px;")
        btn_v6_stop.clicked.connect(self.stop_vision_process)
        top_strip.addWidget(btn_v6_stop)
        l_layout.addLayout(top_strip)

        self.lbl_vision_video_tab6 = QtWidgets.QLabel(
            "Chưa có khung hình D435.\nHãy bật Gazebo (Tab 6) và bấm '▶ Bật Vision'."
        )
        self.lbl_vision_video_tab6.setAlignment(Qt.AlignCenter)
        self.lbl_vision_video_tab6.setMinimumSize(420, 310)
        self.lbl_vision_video_tab6.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.lbl_vision_video_tab6.setStyleSheet(
            "background-color: #05080F; color: #64748B; border: 1px solid #334155; "
            "border-radius: 6px; font-size: 12px; font-weight: bold;"
        )
        l_layout.addWidget(self.lbl_vision_video_tab6, stretch=1)

        # Quick helper + Robotiq 2F-85 Gripper Width & Force Limit Card (synced with Tab 3)
        quick_card = QtWidgets.QFrame()
        quick_card.setObjectName("cardFrame")
        q_vbox = QtWidgets.QVBoxLayout(quick_card)
        q_vbox.setContentsMargins(8, 6, 8, 6)
        q_vbox.setSpacing(5)

        q_top = QtWidgets.QHBoxLayout()
        q_top.setSpacing(6)
        self.chk_vision_sync_obs_tab6 = QtWidgets.QCheckBox("Đồng bộ Cản → NMPC")
        self.chk_vision_sync_obs_tab6.setStyleSheet("color: #34D399; font-weight: bold; font-size: 10px;")
        self.chk_vision_sync_obs_tab6.setFixedWidth(215)
        self.chk_vision_sync_obs_tab6.toggled.connect(self.on_toggle_vision_obs_sync)
        q_top.addWidget(self.chk_vision_sync_obs_tab6)
        q_top.addStretch()

        q_top.addWidget(QtWidgets.QLabel("Bù Z:"))
        self.sp_vpp_pick_dz = QtWidgets.QDoubleSpinBox()
        self.sp_vpp_pick_dz.setRange(-0.045, 0.030)
        self.sp_vpp_pick_dz.setSingleStep(0.002)
        self.sp_vpp_pick_dz.setDecimals(3)
        self.sp_vpp_pick_dz.setValue(-0.018)
        self.sp_vpp_pick_dz.setSuffix(" m")
        self.sp_vpp_pick_dz.setFixedWidth(74)
        self.sp_vpp_pick_dz.setToolTip("Bù khoảng cách từ đầu ngón kẹp lên giữa má kẹp để kẹp đúng 1/2 chiều cao vật")
        q_top.addWidget(self.sp_vpp_pick_dz)

        q_top.addWidget(QtWidgets.QLabel("Delay giữ:"))
        self.sp_vpp_grasp_delay = QtWidgets.QDoubleSpinBox()
        self.sp_vpp_grasp_delay.setRange(0.2, 3.0)
        self.sp_vpp_grasp_delay.setSingleStep(0.1)
        self.sp_vpp_grasp_delay.setDecimals(1)
        self.sp_vpp_grasp_delay.setValue(1.2)
        self.sp_vpp_grasp_delay.setSuffix(" s")
        self.sp_vpp_grasp_delay.setFixedWidth(64)
        self.sp_vpp_grasp_delay.setToolTip("Thời gian chờ ổn định sau khi ngón kẹp đóng xong trước khi nhấc vật lên")
        q_top.addWidget(self.sp_vpp_grasp_delay)

        btn_q_home = QtWidgets.QPushButton("🏠 Về HOME")
        btn_q_home.setStyleSheet("background: #1E3A8A; color: white; font-weight: bold; padding: 4px 8px; font-size: 10px;")
        btn_q_home.clicked.connect(lambda: self.send_joint_pose([0.0, 0.0, 1.5708, 0.0, 1.5708, 0.0]))
        q_top.addWidget(btn_q_home)
        q_vbox.addLayout(q_top)

        # Gripper Width Slider + Presets (synced 2-way with Tab 3)
        gw_row = QtWidgets.QHBoxLayout()
        gw_row.setSpacing(5)
        self.lbl_grip_width_tab6 = QtWidgets.QLabel("🤏 Khẩu độ: 40 mm")
        self.lbl_grip_width_tab6.setStyleSheet("color: #FBBF24; font-weight: bold; font-family: monospace; font-size: 10px;")
        self.lbl_grip_width_tab6.setFixedSize(128, 18)
        self.grip_slider_tab6 = QtWidgets.QSlider(Qt.Horizontal)
        self.grip_slider_tab6.setRange(0, 85)
        self.grip_slider_tab6.setValue(int(self.gripper_width))
        self.grip_slider_tab6.valueChanged.connect(self._on_tab6_gripper_slider)
        gw_row.addWidget(self.lbl_grip_width_tab6)
        gw_row.addWidget(self.grip_slider_tab6, stretch=1)
        for lbl_txt, w_mm, bg_col in [
            ("Mở (85)", 85, None),
            ("Phôi 6cm (55)", 55, "#B45309"),
            ("Chặt (48)", 48, None),
            ("Đóng (0)", 0, None),
        ]:
            b = QtWidgets.QPushButton(lbl_txt)
            b.setStyleSheet(
                f"background: {bg_col}; color: white; padding: 3px 6px; font-size: 10px;"
                if bg_col else "padding: 3px 6px; font-size: 10px;"
            )
            b.clicked.connect(lambda _, val=w_mm: self._set_tab6_grip_preset(val))
            gw_row.addWidget(b)
        q_vbox.addLayout(gw_row)

        # Gripper Force Limit Slider + Presets (synced 2-way with Tab 1)
        gf_row = QtWidgets.QHBoxLayout()
        gf_row.setSpacing(5)
        self.lbl_force_val_tab6 = QtWidgets.QLabel(f"💪 Giới hạn lực: {int(self.gripper_force)} N")
        self.lbl_force_val_tab6.setStyleSheet("color: #38BDF8; font-weight: bold; font-family: monospace; font-size: 10px;")
        self.lbl_force_val_tab6.setFixedSize(128, 18)
        self.force_slider_tab6 = QtWidgets.QSlider(Qt.Horizontal)
        self.force_slider_tab6.setRange(10, 120)
        self.force_slider_tab6.setValue(int(self.gripper_force))
        self.force_slider_tab6.valueChanged.connect(self._on_tab6_force_slider)
        gf_row.addWidget(self.lbl_force_val_tab6)
        gf_row.addWidget(self.force_slider_tab6, stretch=1)
        for f_txt, f_val, f_bg in [("25N", 25, None), ("50N", 50, "#0284C7"), ("90N", 90, None)]:
            bf = QtWidgets.QPushButton(f_txt)
            bf.setStyleSheet(
                f"background: {f_bg}; color: white; padding: 3px 7px; font-size: 10px;"
                if f_bg else "padding: 3px 7px; font-size: 10px;"
            )
            bf.clicked.connect(lambda _, val=f_val: self.force_slider_tab6.setValue(val))
            gf_row.addWidget(bf)
        q_vbox.addLayout(gf_row)

        # Confirm Gripper Width & Force Row
        g_conf_row = QtWidgets.QHBoxLayout()
        g_conf_row.setSpacing(6)
        g_conf_row.addWidget(QtWidgets.QLabel("Độ rộng gắp:"))
        self.sp_vpp_grasp_width = QtWidgets.QSpinBox()
        self.sp_vpp_grasp_width.setRange(0, 78)
        self.sp_vpp_grasp_width.setValue(55)
        self.sp_vpp_grasp_width.setSuffix(" mm")
        self.sp_vpp_grasp_width.setFixedWidth(70)
        self.sp_vpp_grasp_width.valueChanged.connect(self._on_vpp_grasp_spin_changed)
        g_conf_row.addWidget(self.sp_vpp_grasp_width)

        btn_confirm_grip = QtWidgets.QPushButton("✓ Xác nhận mức kẹp")
        btn_confirm_grip.setStyleSheet(
            "QPushButton { background: #059669; color: white; font-weight: bold; padding: 4px 10px; font-size: 10px; border-radius: 5px; }"
            "QPushButton:hover { background: #047857; }"
        )
        btn_confirm_grip.clicked.connect(self.confirm_vpp_grasp_settings)
        g_conf_row.addWidget(btn_confirm_grip)

        self.lbl_vpp_grasp_confirmed = _fix_label(
            QtWidgets.QLabel("⚪ Mặc định 55 mm (Phôi 6cm) | Bấm Xác nhận để chốt"),
            fixed_h=18,
        )
        self.lbl_vpp_grasp_confirmed.setStyleSheet("color: #FBBF24; font-family: monospace; font-size: 10px; font-weight: bold;")
        g_conf_row.addWidget(self.lbl_vpp_grasp_confirmed, stretch=1)
        q_vbox.addLayout(g_conf_row)

        l_layout.addWidget(quick_card)

        layout.addWidget(left_box, stretch=5)

        # ----------------------------------------------------
        # RIGHT COLUMN (stretch=5): 1. Detected Parts & 2x2 Slots | 2. Mode Auto vs Manual & Teach | 3. Run Arc
        # ----------------------------------------------------
        right_col = QtWidgets.QWidget()
        r_outer = QtWidgets.QVBoxLayout(right_col)
        r_outer.setContentsMargins(0, 0, 0, 0)
        r_outer.setSpacing(6)

        # Section 1: Pure Workpiece Recognition & Selection (Camera D435)
        obj_box = QtWidgets.QGroupBox("1. Nhận Diện & Chọn Phôi (Camera D435)")
        ob_lay = QtWidgets.QVBoxLayout(obj_box)
        ob_lay.setContentsMargins(8, 8, 8, 6)
        ob_lay.setSpacing(4)

        obj_row = QtWidgets.QHBoxLayout()
        obj_row.setSpacing(6)

        box_sel_card = QtWidgets.QFrame()
        box_sel_card.setObjectName("cardFrame")
        bsc_lay = QtWidgets.QVBoxLayout(box_sel_card)
        bsc_lay.setContentsMargins(7, 5, 7, 5)
        bsc_lay.setSpacing(3)

        self.lbl_vpp_box_state = _fix_label(
            QtWidgets.QLabel("📡 Đang chờ Camera quét phôi trên băng tải..."),
            fixed_h=18,
        )
        self.lbl_vpp_box_state.setStyleSheet("font-size: 10px; font-weight: bold; color: #94A3B8;")
        bsc_lay.addWidget(self.lbl_vpp_box_state)

        # Orange Cube live row
        self.rb_vpp_orange = QtWidgets.QRadioButton("🟧 Vuông Cam (6×6×6)")
        self.rb_vpp_orange.setStyleSheet("color: #FB923C; font-weight: bold; font-size: 10px;")
        self.rb_vpp_orange.toggled.connect(lambda ck: setattr(self, "vpp_manual_target_pref", "orange_cube") if ck else None)
        self.lbl_vpp_orange_info = _fix_label(QtWidgets.QLabel("   XYZ: --- | ⚪ Chưa thấy"), fixed_h=18)
        self.lbl_vpp_orange_info.setStyleSheet("font-family: monospace; font-size: 10px; color: #FDBA74;")
        bsc_lay.addWidget(self.rb_vpp_orange)
        bsc_lay.addWidget(self.lbl_vpp_orange_info)

        # Green Cylinder live row
        self.rb_vpp_green = QtWidgets.QRadioButton("🟢 Trụ Xanh (Ø6×6)")
        self.rb_vpp_green.setStyleSheet("color: #4ADE80; font-weight: bold; font-size: 10px;")
        self.rb_vpp_green.toggled.connect(lambda ck: setattr(self, "vpp_manual_target_pref", "green_cyl") if ck else None)
        self.lbl_vpp_green_info = _fix_label(QtWidgets.QLabel("   XYZ: --- | ⚪ Chưa thấy"), fixed_h=18)
        self.lbl_vpp_green_info.setStyleSheet("font-family: monospace; font-size: 10px; color: #86EFAC;")
        bsc_lay.addWidget(self.rb_vpp_green)
        bsc_lay.addWidget(self.lbl_vpp_green_info)

        # Active in-zone summary (compact)
        self.lbl_vpp_box_xyz = _fix_label(
            QtWidgets.QLabel("Ô gắp: Chờ phôi vào ranh giới [-0.465..-0.325]"),
            fixed_h=16,
        )
        self.lbl_vpp_box_xyz.setStyleSheet("font-family: monospace; font-size: 9px; color: #38BDF8;")
        bsc_lay.addWidget(self.lbl_vpp_box_xyz)
        obj_row.addWidget(box_sel_card, stretch=3)

        ext_sel_card = QtWidgets.QFrame()
        ext_sel_card.setObjectName("cardFrame")
        esc_lay = QtWidgets.QVBoxLayout(ext_sel_card)
        esc_lay.setContentsMargins(7, 5, 7, 5)
        esc_lay.setSpacing(4)
        self.lbl_vpp_tray_counts = _fix_label(QtWidgets.QLabel("🧩 Ô 2×2 → Cam: 0/2 | Xanh: 0/2"), fixed_h=18)
        self.lbl_vpp_tray_counts.setStyleSheet("color: #FDE047; font-weight: bold; font-family: monospace; font-size: 10px;")
        self.lbl_vpp_obs_summary = _fix_label(QtWidgets.QLabel("🟡 Cản Vàng: --- | Chưa quét"), fixed_h=18)
        self.lbl_vpp_obs_summary.setStyleSheet("font-family: monospace; font-size: 10px; color: #FACC15;")
        self.lbl_vpp_spawn_count = _fix_label(QtWidgets.QLabel("📦 Phôi trên băng tải: 2 gốc (+0 thêm)"), fixed_h=18)
        self.lbl_vpp_spawn_count.setStyleSheet("font-family: monospace; font-size: 10px; color: #94A3B8;")

        esc_lay.addWidget(self.lbl_vpp_tray_counts)
        esc_lay.addWidget(self.lbl_vpp_obs_summary)
        esc_lay.addWidget(self.lbl_vpp_spawn_count)
        esc_lay.addStretch()
        obj_row.addWidget(ext_sel_card, stretch=2)

        ob_lay.addLayout(obj_row)
        r_outer.addWidget(obj_box)

        # Section 2: Mode Selector (Auto Continuous 2x2 vs Manual 1-Shot Free-Drop) + Dobot Jog & Teach
        place_box = QtWidgets.QGroupBox("2. Chế Độ Gắp (Auto Liên Tục / Manual 1 Lần) & Điểm Đặt")
        pl_lay = QtWidgets.QVBoxLayout(place_box)
        pl_lay.setContentsMargins(8, 8, 8, 6)
        pl_lay.setSpacing(5)

        mode_bar = QtWidgets.QHBoxLayout()
        mode_bar.setSpacing(8)
        self.rb_vpp_mode_auto = QtWidgets.QRadioButton("🤖 Auto Liên Tục (Xếp Ô 2×2)")
        self.rb_vpp_mode_auto.setChecked(True)
        self.rb_vpp_mode_auto.setStyleSheet("color: #34D399; font-weight: bold; font-size: 11px;")
        self.rb_vpp_mode_auto.toggled.connect(lambda ck: self._on_vpp_run_mode_changed("auto") if ck else None)
        mode_bar.addWidget(self.rb_vpp_mode_auto)

        self.chk_vpp_autofeed = QtWidgets.QCheckBox("Tự đẩy phôi băng tải")
        self.chk_vpp_autofeed.setChecked(True)
        self.chk_vpp_autofeed.setStyleSheet("color: #86EFAC; font-size: 10px; font-weight: bold;")
        self.chk_vpp_autofeed.toggled.connect(lambda ck: setattr(self, "vpp_conveyor_autofeed", bool(ck)))
        mode_bar.addWidget(self.chk_vpp_autofeed)

        self.rb_vpp_mode_manual = QtWidgets.QRadioButton("🖐️ Manual 1 Lần (Vùng Trống)")
        self.rb_vpp_mode_manual.setStyleSheet("color: #38BDF8; font-weight: bold; font-size: 11px;")
        self.rb_vpp_mode_manual.toggled.connect(lambda ck: self._on_vpp_run_mode_changed("manual") if ck else None)
        mode_bar.addWidget(self.rb_vpp_mode_manual)
        pl_lay.addLayout(mode_bar)

        self.jog_panel_tab6 = self._create_synced_jog_panel(compact=True)
        pl_lay.addWidget(self.jog_panel_tab6)

        teach_row = QtWidgets.QHBoxLayout()
        teach_row.setSpacing(6)
        btn_confirm_place = QtWidgets.QPushButton("📍 XÁC NHẬN VỊ TRÍ ĐẶT (TCP + Góc R)")
        btn_confirm_place.setStyleSheet(
            "QPushButton { background: #0284C7; color: white; font-weight: bold; padding: 6px 10px; border-radius: 6px; }"
            "QPushButton:hover { background: #0369A1; }"
        )
        btn_confirm_place.clicked.connect(self.confirm_vpp_place_pose)
        teach_row.addWidget(btn_confirm_place, stretch=2)

        btn_test_place = QtWidgets.QPushButton("🎯 Tới Thử")
        btn_test_place.setStyleSheet("background: #334155; color: #38BDF8; font-weight: bold; padding: 6px 8px;")
        btn_test_place.clicked.connect(self.preview_vpp_place_pose)
        teach_row.addWidget(btn_test_place)

        btn_clear_place = QtWidgets.QPushButton("🧹 Xóa")
        btn_clear_place.setStyleSheet("background: #1E293B; color: #94A3B8; padding: 6px 8px;")
        btn_clear_place.clicked.connect(self.clear_vpp_place_pose)
        teach_row.addWidget(btn_clear_place)
        pl_lay.addLayout(teach_row)

        place_opt_row = QtWidgets.QHBoxLayout()
        place_opt_row.setSpacing(6)
        self.lbl_vpp_place_status = _fix_label(
            QtWidgets.QLabel("⚪ Chưa xác nhận vị trí đặt — Hãy Jog tay máy đến điểm cần đặt rồi bấm Xác Nhận."),
            fixed_h=28,
        )
        self.lbl_vpp_place_status.setStyleSheet("color: #FBBF24; font-family: monospace; font-size: 10px; font-weight: bold;")
        self.lbl_vpp_place_status.setWordWrap(True)
        place_opt_row.addWidget(self.lbl_vpp_place_status, stretch=2)

        self.cmb_vpp_place_rot = QtWidgets.QComboBox()
        self.cmb_vpp_place_rot.addItems([
            "Góc đặt: Giữ R đã xác nhận",
            "Góc đặt: Đưa R về 0°",
            "Góc đặt: Theo Yaw lúc gắp",
            "Góc đặt: Xoay thêm +90°",
        ])
        self.cmb_vpp_place_rot.setStyleSheet(
            "background: #101726; color: #5EEAD4; border: 1px solid #334155; "
            "border-radius: 4px; padding: 4px 6px; font-size: 10px; font-weight: bold;"
        )
        place_opt_row.addWidget(self.cmb_vpp_place_rot, stretch=1)
        pl_lay.addLayout(place_opt_row)

        r_outer.addWidget(place_box)

        # Section 3: Auto Upward-Arc Pick & Place Execution
        exec_box = QtWidgets.QGroupBox("3. Tự Động Gắp – Vòng Cung Lên – Né Cản – Thả")
        ex_lay = QtWidgets.QVBoxLayout(exec_box)
        ex_lay.setContentsMargins(8, 8, 8, 6)
        ex_lay.setSpacing(6)

        cfg_row = QtWidgets.QHBoxLayout()
        cfg_row.setSpacing(8)
        cfg_row.addWidget(QtWidgets.QLabel("Đỉnh vòng cung tối thiểu (Z):"))
        self.sp_vpp_arc_z = QtWidgets.QDoubleSpinBox()
        self.sp_vpp_arc_z.setRange(0.24, 0.55)
        self.sp_vpp_arc_z.setSingleStep(0.02)
        self.sp_vpp_arc_z.setValue(0.38)
        self.sp_vpp_arc_z.setSuffix(" m")
        cfg_row.addWidget(self.sp_vpp_arc_z)

        self.chk_vpp_return_home = QtWidgets.QCheckBox("Vòng cung về HOME sau khi thả")
        self.chk_vpp_return_home.setChecked(True)
        self.chk_vpp_return_home.setStyleSheet("color: #CBD5E1; font-size: 10px;")
        cfg_row.addWidget(self.chk_vpp_return_home)
        cfg_row.addStretch()
        ex_lay.addLayout(cfg_row)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)
        self.btn_vpp_start = QtWidgets.QPushButton("▶ CHẠY AUTO LIÊN TỤC (XẾP Ô 2×2)")
        self.btn_vpp_start.setStyleSheet(
            "QPushButton { background: #059669; color: white; font-weight: 900; padding: 9px 12px; border-radius: 6px; font-size: 11px; }"
            "QPushButton:hover { background: #047857; }"
            "QPushButton:disabled { background: #1E293B; color: #64748B; }"
        )
        self.btn_vpp_start.clicked.connect(self.start_vision_pick_place)
        btn_row.addWidget(self.btn_vpp_start, stretch=3)

        self.btn_vpp_stop = QtWidgets.QPushButton("⏹ DỪNG")
        self.btn_vpp_stop.setEnabled(False)
        self.btn_vpp_stop.setStyleSheet(
            "QPushButton { background: #DC2626; color: white; font-weight: bold; padding: 9px 14px; border-radius: 6px; }"
            "QPushButton:disabled { background: #1E293B; color: #64748B; }"
        )
        self.btn_vpp_stop.clicked.connect(self.stop_vision_pick_place)
        btn_row.addWidget(self.btn_vpp_stop, stretch=1)
        ex_lay.addLayout(btn_row)

        self.lbl_vpp_phase = _fix_label(
            QtWidgets.QLabel("Sẵn sàng: Chọn vật ở (1), đặt điểm thả ở (2), rồi bấm Bắt Đầu."),
            fixed_h=18,
        )
        self.lbl_vpp_phase.setStyleSheet("color: #94A3B8; font-weight: bold; font-size: 10px;")
        ex_lay.addWidget(self.lbl_vpp_phase)

        self.vpp_progress = QtWidgets.QProgressBar()
        self.vpp_progress.setRange(0, 100)
        self.vpp_progress.setValue(0)
        self.vpp_progress.setTextVisible(False)
        self.vpp_progress.setFixedHeight(8)
        self.vpp_progress.setStyleSheet(
            "QProgressBar { background: #1E293B; border-radius: 4px; } "
            "QProgressBar::chunk { background: #10B981; border-radius: 4px; }"
        )
        ex_lay.addWidget(self.vpp_progress)

        r_outer.addWidget(exec_box)

        # Section 4: Bottom Conveyor Drive (Auto vs Manual Tiến/Lùi) & Workpiece Management (1. Reset | 2. Thêm Phôi)
        conv_box = QtWidgets.QGroupBox("4. Điều Khiển Băng Tải & Quản Lý Phôi (Reset / Thêm Phôi)")
        cb_lay = QtWidgets.QVBoxLayout(conv_box)
        cb_lay.setContentsMargins(8, 8, 8, 6)
        cb_lay.setSpacing(6)

        conv_top = QtWidgets.QHBoxLayout()
        conv_top.setSpacing(6)

        self.conv_ctrl_stack = QtWidgets.QStackedWidget()
        self.conv_ctrl_stack.setFixedHeight(32)

        # Page 0 (Auto): Single smart forward toggle button
        w_auto_conv = QtWidgets.QWidget()
        wac_lay = QtWidgets.QHBoxLayout(w_auto_conv)
        wac_lay.setContentsMargins(0, 0, 0, 0)
        self.btn_vpp_conveyor = QtWidgets.QPushButton("🏭 Chạy Băng Tải (Đưa vào Ô gắp)")
        self.btn_vpp_conveyor.setCheckable(True)
        self.btn_vpp_conveyor.setStyleSheet(
            "QPushButton { background: #0369A1; color: white; font-weight: bold; padding: 5px 10px; border-radius: 5px; font-size: 10px; }"
            "QPushButton:checked { background: #059669; color: #ECFDF5; border: 1px solid #34D399; }"
            "QPushButton:hover { background: #0284C7; }"
        )
        self.btn_vpp_conveyor.clicked.connect(lambda ck: self.toggle_conveyor_feeder(ck, direction=1))
        wac_lay.addWidget(self.btn_vpp_conveyor)
        self.conv_ctrl_stack.addWidget(w_auto_conv)

        # Page 1 (Manual): Lùi | Dừng | Tiến controls
        w_man_conv = QtWidgets.QWidget()
        wmc_lay = QtWidgets.QHBoxLayout(w_man_conv)
        wmc_lay.setContentsMargins(0, 0, 0, 0)
        wmc_lay.setSpacing(5)
        self.btn_vpp_conv_rev = QtWidgets.QPushButton("◀ Lùi Băng Tải")
        self.btn_vpp_conv_rev.setCheckable(True)
        self.btn_vpp_conv_rev.setStyleSheet(
            "QPushButton { background: #7C2D12; color: #FED7AA; font-weight: bold; padding: 5px 8px; border-radius: 5px; font-size: 10px; }"
            "QPushButton:checked { background: #EA580C; color: white; border: 1px solid #FDBA74; }"
        )
        self.btn_vpp_conv_rev.clicked.connect(lambda: self.toggle_conveyor_direction(-1))
        wmc_lay.addWidget(self.btn_vpp_conv_rev, stretch=1)

        self.btn_vpp_conv_stop = QtWidgets.QPushButton("⏹ Dừng")
        self.btn_vpp_conv_stop.setStyleSheet(
            "QPushButton { background: #334155; color: #F8FAFC; font-weight: bold; padding: 5px 10px; border-radius: 5px; font-size: 10px; }"
            "QPushButton:hover { background: #475569; }"
        )
        self.btn_vpp_conv_stop.clicked.connect(lambda: self.toggle_conveyor_feeder(False))
        wmc_lay.addWidget(self.btn_vpp_conv_stop)

        self.btn_vpp_conv_fwd = QtWidgets.QPushButton("▶ Tiến Băng Tải")
        self.btn_vpp_conv_fwd.setCheckable(True)
        self.btn_vpp_conv_fwd.setStyleSheet(
            "QPushButton { background: #0369A1; color: white; font-weight: bold; padding: 5px 8px; border-radius: 5px; font-size: 10px; }"
            "QPushButton:checked { background: #059669; color: #ECFDF5; border: 1px solid #34D399; }"
        )
        self.btn_vpp_conv_fwd.clicked.connect(lambda: self.toggle_conveyor_direction(1))
        wmc_lay.addWidget(self.btn_vpp_conv_fwd, stretch=1)
        self.conv_ctrl_stack.addWidget(w_man_conv)

        conv_top.addWidget(self.conv_ctrl_stack, stretch=2)

        self.lbl_vpp_conv_dir = _fix_label(QtWidgets.QLabel("⏹ Băng tải: Đang dừng"), fixed_h=18)
        self.lbl_vpp_conv_dir.setFixedWidth(150)
        self.lbl_vpp_conv_dir.setStyleSheet("font-family: monospace; font-size: 10px; font-weight: bold; color: #94A3B8;")
        conv_top.addWidget(self.lbl_vpp_conv_dir)
        cb_lay.addLayout(conv_top)

        # Row 2: 2 options (1. Reset [giữ 2 phôi gốc] | 2. Thêm Phôi Vuông/Trụ ở đầu kia băng tải)
        part_row = QtWidgets.QHBoxLayout()
        part_row.setSpacing(6)
        self.cmb_spawn_shape = QtWidgets.QComboBox()
        self.cmb_spawn_shape.addItems([
            "🟧 Thêm: Vuông Cam (6×6×6)",
            "🟢 Thêm: Trụ Xanh (Ø6×6)",
        ])
        self.cmb_spawn_shape.setStyleSheet(
            "background: #101726; color: #F8FAFC; border: 1px solid #334155; "
            "border-radius: 4px; padding: 4px 6px; font-size: 10px; font-weight: bold;"
        )
        part_row.addWidget(self.cmb_spawn_shape, stretch=2)

        self.btn_vpp_add_part = QtWidgets.QPushButton("➕ 2. Thêm Phôi")
        self.btn_vpp_add_part.setToolTip("Thả thêm phôi mới ở phía đầu vào băng tải (cao ≥ 2× chiều cao phôi để tránh chồng)")
        self.btn_vpp_add_part.setStyleSheet(
            "QPushButton { background: #1D4ED8; color: white; font-weight: bold; padding: 5px 10px; border-radius: 5px; font-size: 10px; }"
            "QPushButton:hover { background: #2563EB; }"
        )
        self.btn_vpp_add_part.clicked.connect(lambda: self.spawn_additional_workpiece())
        part_row.addWidget(self.btn_vpp_add_part, stretch=1)

        self.btn_vpp_reset_parts = QtWidgets.QPushButton("🔄 1. Reset Phôi")
        self.btn_vpp_reset_parts.setToolTip("Xóa toàn bộ phôi đã thêm và đặt lại đúng 2 phôi gốc ban đầu + làm mới Ô 2×2")
        self.btn_vpp_reset_parts.setStyleSheet(
            "QPushButton { background: #0F766E; color: white; font-weight: bold; padding: 5px 10px; border-radius: 5px; font-size: 10px; }"
            "QPushButton:hover { background: #0D9488; }"
        )
        self.btn_vpp_reset_parts.clicked.connect(self.reset_target_box_pose)
        part_row.addWidget(self.btn_vpp_reset_parts, stretch=1)
        cb_lay.addLayout(part_row)

        r_outer.addWidget(conv_box)
        r_outer.addStretch()
        layout.addWidget(right_col, stretch=5)
        return w

    # ========================================================
    # TAB 4: NMPC & OBSTACLE SETUP (EDITABLE PARAMETERS)
    # ========================================================
    def create_nmpc_tab(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        layout.setSpacing(12)

        # Top Apply Bar
        top_bar = QtWidgets.QFrame()
        top_bar.setObjectName("cardFrame")
        tb_layout = QtWidgets.QHBoxLayout(top_bar)
        
        desc = QtWidgets.QLabel("Chỉnh sửa trực tiếp thông số vật cản 3D thực tế và trọng số acados NMPC:")
        desc.setStyleSheet("color: #94A3B8; font-size: 12px;")
        btn_apply = QtWidgets.QPushButton("LƯU & ÁP DỤNG THÔNG SỐ (APPLY)")
        btn_apply.setStyleSheet("background: #059669; color: white; font-weight: bold; padding: 8px 18px;")
        btn_apply.clicked.connect(self.apply_nmpc_params)

        tb_layout.addWidget(desc)
        tb_layout.addStretch()
        tb_layout.addWidget(btn_apply)
        layout.addWidget(top_bar)

        # Two Column Forms
        form_layout = QtWidgets.QHBoxLayout()

        # Panel 1: Obstacle Coordinates
        obs_box = QtWidgets.QGroupBox("1. Tọa độ Vật cản & Vùng An Toàn 3D (m / cm)")
        ob_layout = QtWidgets.QFormLayout(obs_box)
        ob_layout.setSpacing(10)

        self.sp_obs_x = QtWidgets.QDoubleSpinBox()
        self.sp_obs_x.setRange(-1.0, 1.0)
        self.sp_obs_x.setValue(0.45)
        self.sp_obs_x.setSingleStep(0.01)

        self.sp_obs_y = QtWidgets.QDoubleSpinBox()
        self.sp_obs_y.setRange(-1.0, 1.0)
        self.sp_obs_y.setValue(0.00)
        self.sp_obs_y.setSingleStep(0.01)

        self.sp_obs_z = QtWidgets.QDoubleSpinBox()
        self.sp_obs_z.setRange(0.0, 1.0)
        self.sp_obs_z.setValue(0.15)
        self.sp_obs_z.setSingleStep(0.01)

        self.sp_obs_r = QtWidgets.QSpinBox()
        self.sp_obs_r.setRange(2, 25)
        self.sp_obs_r.setValue(7) # 7cm

        self.sp_warn_m = QtWidgets.QSpinBox()
        self.sp_warn_m.setRange(5, 30)
        self.sp_warn_m.setValue(15) # 15cm

        self.sp_danger_m = QtWidgets.QSpinBox()
        self.sp_danger_m.setRange(1, 10)
        self.sp_danger_m.setValue(1) # 1cm (per user request!)

        ob_layout.addRow("Tọa độ X trụ (m):", self.sp_obs_x)
        ob_layout.addRow("Tọa độ Y trụ (m):", self.sp_obs_y)
        ob_layout.addRow("Tọa độ Z trụ (m):", self.sp_obs_z)
        ob_layout.addRow("Bán kính trụ r (cm):", self.sp_obs_r)
        ob_layout.addRow("Vùng đệm né cản (cm):", self.sp_warn_m)
        ob_layout.addRow("Ngưỡng AUTO E-STOP (cm):", self.sp_danger_m)

        form_layout.addWidget(obs_box)

        # Panel 2: acados NMPC Weights
        nmpc_box = QtWidgets.QGroupBox("2. Trọng số Tối ưu hóa acados NMPC")
        nb_layout = QtWidgets.QFormLayout(nmpc_box)
        nb_layout.setSpacing(10)

        self.sp_w_pos = QtWidgets.QDoubleSpinBox()
        self.sp_w_pos.setRange(10.0, 500.0)
        self.sp_w_pos.setValue(180.0)

        self.sp_w_ori = QtWidgets.QDoubleSpinBox()
        self.sp_w_ori.setRange(1.0, 200.0)
        self.sp_w_ori.setValue(50.0)

        self.sp_w_obs = QtWidgets.QDoubleSpinBox()
        self.sp_w_obs.setRange(10.0, 500.0)
        self.sp_w_obs.setValue(100.0)

        self.sp_horizon = QtWidgets.QSpinBox()
        self.sp_horizon.setRange(5, 40)
        self.sp_horizon.setValue(20)

        self.sp_clamp = QtWidgets.QDoubleSpinBox()
        self.sp_clamp.setRange(0.5, 5.0)
        self.sp_clamp.setValue(1.4)

        nb_layout.addRow("W_pos (Bám vị trí):", self.sp_w_pos)
        nb_layout.addRow("W_ori (Giữ mỏ kẹp đứng):", self.sp_w_ori)
        nb_layout.addRow("W_obs (Đẩy né cản):", self.sp_w_obs)
        nb_layout.addRow("Horizon Steps N:", self.sp_horizon)
        nb_layout.addRow("Max Joint Disp (°/20ms):", self.sp_clamp)

        form_layout.addWidget(nmpc_box)

        layout.addLayout(form_layout)
        layout.addStretch()
        return w

    # ========================================================
    # TAB 5: PROCESS SUPERVISOR (ONE-CLICK LAUNCHER)
    # ========================================================
    def create_launcher_tab(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        layout.setSpacing(12)

        box = QtWidgets.QGroupBox("ROS 2 Process Launcher (Khởi động hệ thống không cần terminal)")
        b_layout = QtWidgets.QVBoxLayout(box)
        b_layout.setSpacing(12)

        # Row 1: Gazebo Sim
        row1 = QtWidgets.QHBoxLayout()
        self.lbl_gz_world_desc = QtWidgets.QLabel("1. Gazebo Harmonic Sim (pick_and_place_rail.sdf + ros2_control)")
        self.lbl_gz_world_desc.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.lbl_gz_status = QtWidgets.QLabel("🔴 ĐÃ DỪNG")
        self.lbl_gz_status.setStyleSheet("color: #EF4444; font-weight: bold; font-family: monospace; font-size: 11px;")
        
        self.btn_start_gz = QtWidgets.QPushButton("Start Gazebo")
        self.btn_start_gz.setStyleSheet("background: #059669; color: white; font-weight: bold; padding: 6px 14px;")
        self.btn_start_gz.clicked.connect(lambda: self.start_gazebo_process(show_popup=True))
        self.btn_stop_gz = QtWidgets.QPushButton("Stop")
        self.btn_stop_gz.setStyleSheet("background: #7F1D1D; color: white; padding: 6px 14px;")
        self.btn_stop_gz.clicked.connect(self.stop_gazebo_process)
        
        row1.addWidget(self.lbl_gz_world_desc)
        row1.addSpacing(12)
        row1.addWidget(self.lbl_gz_status)
        row1.addStretch()
        row1.addWidget(self.btn_start_gz)
        row1.addWidget(self.btn_stop_gz)
        b_layout.addLayout(row1)

        # Row 2: NMPC Controller
        row2 = QtWidgets.QHBoxLayout()
        r2_lbl = QtWidgets.QLabel("2. C++ acados NMPC Controller Node (50Hz RTI)")
        r2_lbl.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.lbl_nmpc_status = QtWidgets.QLabel("🔴 ĐÃ DỪNG")
        self.lbl_nmpc_status.setStyleSheet("color: #EF4444; font-weight: bold; font-family: monospace; font-size: 11px;")
        
        self.btn_start_nmpc = QtWidgets.QPushButton("Start NMPC")
        self.btn_start_nmpc.setStyleSheet("background: #059669; color: white; font-weight: bold; padding: 6px 14px;")
        self.btn_start_nmpc.clicked.connect(self.start_nmpc_process)
        self.btn_stop_nmpc = QtWidgets.QPushButton("Stop")
        self.btn_stop_nmpc.setStyleSheet("background: #7F1D1D; color: white; padding: 6px 14px;")
        self.btn_stop_nmpc.clicked.connect(self.stop_nmpc_process)
        
        row2.addWidget(r2_lbl)
        row2.addSpacing(12)
        row2.addWidget(self.lbl_nmpc_status)
        row2.addStretch()
        row2.addWidget(self.btn_start_nmpc)
        row2.addWidget(self.btn_stop_nmpc)
        b_layout.addLayout(row2)

        # Row 3: D435 RGB-D Vision Node
        row3 = QtWidgets.QHBoxLayout()
        r3_lbl = QtWidgets.QLabel("3. D435 RGB-D Vision Node (OpenCV HSV + 3D TF2)")
        r3_lbl.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.lbl_vision_status = QtWidgets.QLabel("🔴 ĐÃ DỪNG")
        self.lbl_vision_status.setStyleSheet("color: #EF4444; font-weight: bold; font-family: monospace; font-size: 11px;")

        self.btn_start_vision = QtWidgets.QPushButton("Start Vision")
        self.btn_start_vision.setStyleSheet("background: #059669; color: white; font-weight: bold; padding: 6px 14px;")
        self.btn_start_vision.clicked.connect(self.start_vision_process)
        self.btn_stop_vision = QtWidgets.QPushButton("Stop")
        self.btn_stop_vision.setStyleSheet("background: #7F1D1D; color: white; padding: 6px 14px;")
        self.btn_stop_vision.clicked.connect(self.stop_vision_process)

        row3.addWidget(r3_lbl)
        row3.addSpacing(12)
        row3.addWidget(self.lbl_vision_status)
        row3.addStretch()
        row3.addWidget(self.btn_start_vision)
        row3.addWidget(self.btn_stop_vision)
        b_layout.addLayout(row3)

        # Row 4: Dynamic Obstacle Generator Node
        row4 = QtWidgets.QHBoxLayout()
        r4_lbl = QtWidgets.QLabel("4. Dynamic Obstacle Generator (Hình Tròn / Zigzag khó đoán)")
        r4_lbl.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.lbl_dyn_status = QtWidgets.QLabel("🔴 ĐÃ DỪNG")
        self.lbl_dyn_status.setStyleSheet("color: #EF4444; font-weight: bold; font-family: monospace; font-size: 11px;")

        self.btn_start_dyn = QtWidgets.QPushButton("Start Dynamic")
        self.btn_start_dyn.setStyleSheet("background: #D97706; color: white; font-weight: bold; padding: 6px 14px;")
        self.btn_start_dyn.clicked.connect(self.start_dynamic_obstacle_process)
        self.btn_stop_dyn = QtWidgets.QPushButton("Stop")
        self.btn_stop_dyn.setStyleSheet("background: #7F1D1D; color: white; padding: 6px 14px;")
        self.btn_stop_dyn.clicked.connect(self.stop_dynamic_obstacle_process)

        row4.addWidget(r4_lbl)
        row4.addSpacing(12)
        row4.addWidget(self.lbl_dyn_status)
        row4.addStretch()
        row4.addWidget(self.btn_start_dyn)
        row4.addWidget(self.btn_stop_dyn)
        b_layout.addLayout(row4)

        layout.addWidget(box)
        layout.addStretch()
        return w

    # ========================================================
    # LOGIC & SLOTS
    # ========================================================
    def system_tick(self):
        """Runs at 50Hz: spins ROS, animates execution, updates UI"""
        self.ros.spin_once()

        # 1. Update Tab 5 live process status badges & button enabled state
        gz_proc = self.processes.get('gazebo')
        if gz_proc and gz_proc.poll() is None:
            self.lbl_gz_status.setText(f"🟢 ĐANG CHẠY (PID: {gz_proc.pid})")
            self.lbl_gz_status.setStyleSheet("color: #10B981; font-weight: bold; font-family: monospace;")
            self.btn_start_gz.setEnabled(False)
            self.btn_stop_gz.setEnabled(True)
        else:
            self.lbl_gz_status.setText("🔴 ĐÃ DỪNG")
            self.lbl_gz_status.setStyleSheet("color: #EF4444; font-weight: bold; font-family: monospace;")
            self.btn_start_gz.setEnabled(True)
            self.btn_stop_gz.setEnabled(False)

        nmpc_proc = self.processes.get('nmpc')
        if nmpc_proc and nmpc_proc.poll() is None:
            self.lbl_nmpc_status.setText(f"🟢 ĐANG CHẠY (PID: {nmpc_proc.pid})")
            self.lbl_nmpc_status.setStyleSheet("color: #10B981; font-weight: bold; font-family: monospace;")
            self.btn_start_nmpc.setEnabled(False)
            self.btn_stop_nmpc.setEnabled(True)
        else:
            self.lbl_nmpc_status.setText("🔴 ĐÃ DỪNG")
            self.lbl_nmpc_status.setStyleSheet("color: #EF4444; font-weight: bold; font-family: monospace;")
            self.btn_start_nmpc.setEnabled(True)
            self.btn_stop_nmpc.setEnabled(False)

        vis_proc = self.processes.get('vision')
        if vis_proc and vis_proc.poll() is None:
            self.lbl_vision_status.setText(f"🟢 ĐANG CHẠY (PID: {vis_proc.pid})")
            self.lbl_vision_status.setStyleSheet("color: #10B981; font-weight: bold; font-family: monospace;")
            self.btn_start_vision.setEnabled(False)
            self.btn_stop_vision.setEnabled(True)
        else:
            self.lbl_vision_status.setText("🔴 ĐÃ DỪNG")
            self.lbl_vision_status.setStyleSheet("color: #EF4444; font-weight: bold; font-family: monospace;")
            self.btn_start_vision.setEnabled(True)
            self.btn_stop_vision.setEnabled(False)

        dyn_proc = self.processes.get('dynamic_obs')
        if hasattr(self, "lbl_dyn_status"):
            if dyn_proc and dyn_proc.poll() is None:
                self.lbl_dyn_status.setText(f"🟢 ĐANG CHẠY ({self.dyn_obs_mode.upper()} | PID: {dyn_proc.pid})")
                self.lbl_dyn_status.setStyleSheet("color: #FDBA74; font-weight: bold; font-family: monospace;")
                self.btn_start_dyn.setEnabled(False)
                self.btn_stop_dyn.setEnabled(True)
            else:
                self.lbl_dyn_status.setText("🔴 ĐÃ DỪNG")
                self.lbl_dyn_status.setStyleSheet("color: #EF4444; font-weight: bold; font-family: monospace;")
                self.btn_start_dyn.setEnabled(True)
                self.btn_stop_dyn.setEnabled(False)

        # 2. Autonomous Step Sequencer Engine
        if self.is_running and not self.is_paused and len(self.program_steps) > 0:
            self.cycle_time += 0.02
            self.step_timer += 0.02
            mins = int(self.cycle_time // 60)
            secs = self.cycle_time % 60
            self.lbl_timer.setText(f"Timer: {mins:02d}:{secs:04.1f}s")

            step = self.program_steps[self.current_step_idx]
            self.lbl_prog_step.setText(f"Step {self.current_step_idx + 1}/{len(self.program_steps)}: {step['type']} ({step['target']})")

            step_finished = False
            t = step['type']

            if t == 'MoveNMPC':
                # Check 3D distance between TCP and target
                tx, ty, tz = self.current_target_xyz
                dist = math.hypot(math.hypot(self.tcp_pos[0] - tx, self.tcp_pos[1] - ty), self.tcp_pos[2] - tz)
                if dist < 0.040 or self.step_timer > 6.0:
                    self.settle_timer += 0.02
                    if self.settle_timer >= 0.20:
                        step_finished = True

            elif t == 'MoveJ':
                if self.step_timer >= 1.5:
                    step_finished = True

            elif t == 'Gripper':
                if self.step_timer >= 1.0:
                    step_finished = True

            elif t == 'Wait':
                if self.step_timer >= getattr(self, 'wait_duration', 1.0):
                    step_finished = True

            elif t == 'RotateCircle':
                # Generate intermediate circle waypoints smoothly at configured speed
                # Timer runs at 50Hz (dt = 0.02s)
                d_theta = getattr(self, 'orbit_speed', 60.0) * 0.02
                self.orbit_traveled_angle += d_theta
                current_deg = self.orbit_start_angle + self.orbit_traveled_angle
                rad = math.radians(current_deg)
                cx, cy, cz = self.orbit_center
                ox = cx + self.orbit_radius * math.cos(rad)
                oy = cy + self.orbit_radius * math.sin(rad)
                self.ros.publish_target(ox, oy, cz)

                max_duration = (self.orbit_target_angle / max(10.0, getattr(self, 'orbit_speed', 60.0))) + 3.0
                if self.orbit_traveled_angle >= self.orbit_target_angle or self.step_timer > max_duration:
                    step_finished = True

            if step_finished:
                self.step_in_progress = False
                if self.is_auto_mode:
                    self.current_step_idx += 1
                    if self.current_step_idx >= len(self.program_steps):
                        self.current_step_idx = 0
                        self.is_running = False
                        self.lbl_prog_step.setText("✓ Đã hoàn thành toàn bộ chu trình!")
                    else:
                        self.execute_current_step()
                    self.update_tracker_display()
                else:
                    self.lbl_prog_step.setText(f"✓ Xong Step {self.current_step_idx + 1} - Bấm CHECK GO để sang bước tiếp")
                    self.update_tracker_display()

        # 2b. Autonomous Vision Pick & Place Engine + Conveyor Belt Feeder
        self._step_conveyor_feeder()
        if self.vpp_active:
            self.step_vision_pick_place()
        elif getattr(self, "vpp_auto_armed", False):
            self._step_vpp_auto_monitor()

        # Telemetry updates
        self.lbl_tele_xyz.setText(f"TCP XYZ: [{self.tcp_pos[0]:.3f}, {self.tcp_pos[1]:.3f}, {self.tcp_pos[2]:.3f}] m")
        self.lbl_palette_tcp.setText(f"Tọa độ hiện tại: [{self.tcp_pos[0]:.3f}, {self.tcp_pos[1]:.3f}, {self.tcp_pos[2]:.3f}] m")
        r_deg_live = math.degrees(self.joint_angles[5]) if len(self.joint_angles) >= 6 else math.degrees(self.desired_j6_rad)
        for jp in self.jog_panels:
            jp.update_pose(self.tcp_pos[0], self.tcp_pos[1], self.tcp_pos[2], r_deg_live)
        
        # 3D Distance calculation (NO false collisions when lifting above pillar!)
        dist_3d, delta_r, delta_z = self.radar.compute_3d_distance()
        dist_cm = dist_3d * 100.0
        self.update_safety_display(dist_cm, delta_r * 100.0, delta_z * 100.0)

    def on_joint_state(self, joints):
        self.joint_angles = joints
        for i in range(min(6, len(joints))):
            deg = math.degrees(joints[i])
            self.joint_labels[i].setText(f"J{i+1}: {deg:.1f}°")
            self.joint_sliders[i].blockSignals(True)
            self.joint_sliders[i].setValue(int(deg))
            self.joint_sliders[i].blockSignals(False)

    def on_ee_pose(self, pos):
        self.tcp_pos = pos
        self.radar.update_tcp(pos[0], pos[1], pos[2])

    def on_radar_click(self, wx, wy):
        """When user clicks on 2D table radar, send target to robot"""
        self.tcp_pos[0] = wx
        self.tcp_pos[1] = wy
        self.ros.publish_target(wx, wy, self.tcp_pos[2])
        self.radar.update_tcp(wx, wy, self.tcp_pos[2])

    def _create_synced_jog_panel(self, compact=False):
        panel = DobotJogPanel(initial_vel=self.speed_slider.value(), compact=compact)
        panel.jog_xyz_signal.connect(self.continuous_jog)
        panel.jog_r_signal.connect(self.jog_r_nmpc)
        panel.set_r_abs_signal.connect(self.set_r_abs_nmpc)
        panel.align_box_yaw_signal.connect(self.align_r_to_vision_box)
        panel.velocity_changed_signal.connect(self._on_panel_speed_changed)
        self.jog_panels.append(panel)
        return panel

    def _on_global_speed_changed(self, val):
        self.speed_val_lbl.setText(f"{val}%")
        for jp in self.jog_panels:
            jp.set_velocity_silent(val)

    def _on_panel_speed_changed(self, val):
        self.speed_slider.blockSignals(True)
        self.speed_slider.setValue(int(val))
        self.speed_slider.blockSignals(False)
        self.speed_val_lbl.setText(f"{int(val)}%")
        for jp in self.jog_panels:
            jp.set_velocity_silent(val)

    def _compute_pick_j6(self, bx, by, yaw_box_rad, kind="orange_cube", dims=None):
        """Compute wrist J6 for grasping: flat-face align for boxes (cube 4-way 90deg sym), 0 for round cylinders."""
        if kind == "green_cyl":
            return 0.0
        j1_approx = math.atan2(by, bx)
        j6 = j1_approx - float(yaw_box_rad)
        is_square = (kind == "orange_cube") or (
            dims is not None and len(dims) >= 2 and abs(float(dims[0]) - float(dims[1])) < 0.015
        )
        if is_square:
            while j6 > math.pi * 0.25:
                j6 -= math.pi * 0.5
            while j6 < -math.pi * 0.25:
                j6 += math.pi * 0.5
        else:
            while j6 > math.pi * 0.5:
                j6 -= math.pi
            while j6 < -math.pi * 0.5:
                j6 += math.pi
        return j6

    def jog_r_nmpc(self, deg_delta):
        speed_factor = self.speed_slider.value() / 100.0
        # For discrete quick steps (+45 / -45 / +90), apply full angle; for hold steps (+/-8), scale by velocity
        scale = 1.0 if abs(deg_delta) >= 30.0 else speed_factor
        self.desired_j6_rad = max(-math.pi, min(math.pi, self.desired_j6_rad + math.radians(deg_delta) * scale))
        self.ros.publish_target(self.tcp_pos[0], self.tcp_pos[1], self.tcp_pos[2], yaw=self.desired_j6_rad)

    def set_r_abs_nmpc(self, deg_val):
        self.desired_j6_rad = max(-math.pi, min(math.pi, math.radians(float(deg_val))))
        self.ros.publish_target(self.tcp_pos[0], self.tcp_pos[1], self.tcp_pos[2], yaw=self.desired_j6_rad)

    def align_r_to_vision_box(self):
        box = self._require_vision_box()
        if box is None:
            return
        bx, by, _, yaw_box = box
        kind = getattr(self, "vpp_active_target_kind", None) or "orange_cube"
        dims = getattr(self, "vision_box_dims", None)
        j6 = self._compute_pick_j6(bx, by, yaw_box, kind=kind, dims=dims)
        self.desired_j6_rad = j6
        self.ros.publish_target(self.tcp_pos[0], self.tcp_pos[1], self.tcp_pos[2], yaw=j6)

    def continuous_jog(self, dx_base, dy_base, dz_base):
        """Smooth jog scaled by Speed Override slider (runs continuously on button hold)"""
        speed_factor = self.speed_slider.value() / 100.0
        self.tcp_pos[0] = max(0.15, min(0.78, self.tcp_pos[0] + dx_base * speed_factor))
        self.tcp_pos[1] = max(-0.60, min(0.60, self.tcp_pos[1] + dy_base * speed_factor))
        self.tcp_pos[2] = max(0.018, min(0.85, self.tcp_pos[2] + dz_base * speed_factor))
        self.ros.publish_target(self.tcp_pos[0], self.tcp_pos[1], self.tcp_pos[2], yaw=self.desired_j6_rad)
        self.radar.update_tcp(self.tcp_pos[0], self.tcp_pos[1], self.tcp_pos[2])

    def jog_joint_fine(self, idx, deg_step):
        speed_factor = self.speed_slider.value() / 100.0
        curr_deg = math.degrees(self.joint_angles[idx])
        new_deg = curr_deg + deg_step * speed_factor
        self.on_joint_slider(idx, new_deg)

    def send_joint_pose(self, joint_rads):
        self.joint_angles = list(joint_rads)
        if len(self.joint_angles) >= 6:
            self.desired_j6_rad = float(self.joint_angles[5])
        self.ros.publish_joints(self.joint_angles)

    def on_gripper_slider(self, val):
        self.gripper_width = float(val)
        self.gripper_viz.set_width(val)
        
        # Status text update
        if 32.0 <= val <= 48.0:
            status_text = "🟢 TRẠNG THÁI: ĐANG KẸP HỘP (Hộp 4cm kẹp chặt - Lực: 45N)"
        elif val >= 80.0:
            status_text = "🔵 TRẠNG THÁI: MỞ HOÀN TOÀN (85mm)"
        elif val <= 5.0:
            status_text = "⚪ TRẠNG THÁI: ĐÓNG HẾT / TRỐNG (0mm)"
        else:
            status_text = f"🟡 TRẠNG THÁI: VỊ TRÍ TÙY CHỈNH ({val:.0f}mm)"

        self.lbl_grip_status.setText(status_text)
        self.lbl_tele_grip.setText(f"Robotiq 2F-85: {val}mm")
        if hasattr(self, "grip_slider_tab6") and self.grip_slider_tab6.value() != int(val):
            self.grip_slider_tab6.blockSignals(True)
            self.grip_slider_tab6.setValue(int(val))
            self.grip_slider_tab6.blockSignals(False)
        if hasattr(self, "lbl_grip_width_tab6"):
            self.lbl_grip_width_tab6.setText(f"🤏 Khẩu độ: {int(val)} mm")
        self.ros.publish_gripper(val)

    def _on_tab6_gripper_slider(self, val):
        self.grip_slider.setValue(int(val))
        if not self.vpp_grasp_locked and int(val) <= 78 and hasattr(self, "sp_vpp_grasp_width"):
            self.sp_vpp_grasp_width.blockSignals(True)
            self.sp_vpp_grasp_width.setValue(int(val))
            self.sp_vpp_grasp_width.blockSignals(False)
            if hasattr(self, "lbl_vpp_grasp_confirmed"):
                self.lbl_vpp_grasp_confirmed.setText(
                    f"🟡 Tạm theo thanh trượt: {int(val)} mm | Bấm 'Xác nhận mức kẹp' để chốt"
                )
                self.lbl_vpp_grasp_confirmed.setStyleSheet("color: #FBBF24; font-family: monospace; font-size: 10px; font-weight: bold;")

    def _set_tab6_grip_preset(self, val_mm):
        self.grip_slider.setValue(int(val_mm))
        if val_mm <= 78 and hasattr(self, "sp_vpp_grasp_width"):
            self.sp_vpp_grasp_width.setValue(int(val_mm))

    def _on_vpp_grasp_spin_changed(self, val):
        self.vpp_confirmed_grasp_width = int(val)
        self.vpp_confirmed_grasp_force = int(self.force_slider_tab6.value()) if hasattr(self, "force_slider_tab6") else int(self.gripper_force)
        self.vpp_grasp_locked = True
        if hasattr(self, "lbl_vpp_grasp_confirmed"):
            self.lbl_vpp_grasp_confirmed.setText(
                f"🟢 Đã chốt kẹp: {self.vpp_confirmed_grasp_width} mm | Lực: {self.vpp_confirmed_grasp_force} N"
            )
            self.lbl_vpp_grasp_confirmed.setStyleSheet("color: #34D399; font-family: monospace; font-size: 10px; font-weight: bold;")

    def confirm_vpp_grasp_settings(self):
        w = int(self.grip_slider_tab6.value()) if hasattr(self, "grip_slider_tab6") else int(self.gripper_width)
        w = min(78, max(0, w))
        f = int(self.force_slider_tab6.value()) if hasattr(self, "force_slider_tab6") else int(self.gripper_force)
        self.vpp_confirmed_grasp_width = w
        self.vpp_confirmed_grasp_force = f
        self.vpp_grasp_locked = True
        if hasattr(self, "sp_vpp_grasp_width"):
            self.sp_vpp_grasp_width.blockSignals(True)
            self.sp_vpp_grasp_width.setValue(w)
            self.sp_vpp_grasp_width.blockSignals(False)
        if hasattr(self, "lbl_vpp_grasp_confirmed"):
            self.lbl_vpp_grasp_confirmed.setText(
                f"🟢 Đã chốt kẹp: {w} mm | Lực: {f} N"
            )
            self.lbl_vpp_grasp_confirmed.setStyleSheet("color: #34D399; font-family: monospace; font-size: 10px; font-weight: bold;")

    def _on_tab6_force_slider(self, val):
        self.force_slider.setValue(int(val))
        if self.vpp_grasp_locked:
            self.vpp_confirmed_grasp_force = int(val)
            if hasattr(self, "lbl_vpp_grasp_confirmed") and self.vpp_confirmed_grasp_width is not None:
                self.lbl_vpp_grasp_confirmed.setText(
                    f"🟢 Đã chốt kẹp: {self.vpp_confirmed_grasp_width} mm | Lực: {int(val)} N"
                )

    def on_joint_slider(self, idx, deg_val):
        rad = math.radians(deg_val)
        self.joint_angles[idx] = rad
        self.joint_labels[idx].setText(f"J{idx+1}: {deg_val:.1f}°")
        self.ros.publish_joints(self.joint_angles)

    def fill_target_with_current(self):
        self.inp_target_x.setValue(self.tcp_pos[0])
        self.inp_target_y.setValue(self.tcp_pos[1])
        self.inp_target_z.setValue(self.tcp_pos[2])

    def move_to_input_target(self):
        tx = self.inp_target_x.value()
        ty = self.inp_target_y.value()
        tz = self.inp_target_z.value()
        self.tcp_pos = [tx, ty, tz]
        self.ros.publish_target(tx, ty, tz)
        self.radar.update_tcp(tx, ty, tz)

    def update_safety_display(self, dist_cm, delta_r_cm, delta_z_cm):
        self.dist_val_lbl.setText(f"{dist_cm:.1f} cm")
        self.dist_bar.setValue(int(min(30, dist_cm)))
        
        self.lbl_delta_z.setText(f"Độ cao trên đỉnh cột (Δz): {'+' if delta_z_cm>=0 else ''}{delta_z_cm:.1f} cm")
        self.lbl_delta_r.setText(f"Khoảng cách ngang (Δr): +{delta_r_cm:.1f} cm")

        # Thresholds: Stop at 1.0 cm, Warn at 15.0 cm
        danger_th = self.danger_margin * 100.0 # 1.0 cm
        warn_th = self.warn_margin * 100.0     # 15.0 cm

        if dist_cm <= danger_th:
            self.dist_val_lbl.setStyleSheet("font-size: 18px; font-weight: 900; color: #EF4444; font-family: monospace;")
            self.dist_bar.setStyleSheet("QProgressBar { background: #1E293B; height: 12px; } QProgressBar::chunk { background: #EF4444; }")
            self.safety_desc.setText("⚠️ NGUY HIỂM: Khoảng cách ≤ 1.0cm! AUTO E-STOP ĐÃ KÍCH HOẠT")
            self.safety_desc.setStyleSheet("color: #EF4444; font-weight: bold; font-size: 11px;")
            self.safety_badge.setText("COLLISION STOP")
            self.safety_badge.setStyleSheet("background: #7F1D1D; color: #FCA5A5; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 10px;")
            if self.is_running:
                self.trigger_estop()
        elif dist_cm <= warn_th:
            self.dist_val_lbl.setStyleSheet("font-size: 18px; font-weight: 900; color: #F59E0B; font-family: monospace;")
            self.dist_bar.setStyleSheet("QProgressBar { background: #1E293B; height: 12px; } QProgressBar::chunk { background: #F59E0B; }")
            self.safety_desc.setText("⚡ CẢNH BÁO: Đang trong vùng cản! NMPC kích hoạt lực đẩy né tránh.")
            self.safety_desc.setStyleSheet("color: #F59E0B; font-weight: bold; font-size: 11px;")
            self.safety_badge.setText("NMPC REPEL")
            self.safety_badge.setStyleSheet("background: #78350F; color: #FCD34D; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 10px;")
        else:
            self.dist_val_lbl.setStyleSheet("font-size: 18px; font-weight: 900; color: #34D399; font-family: monospace;")
            self.dist_bar.setStyleSheet("QProgressBar { background: #1E293B; height: 12px; } QProgressBar::chunk { background: #10B981; }")
            self.safety_desc.setText("✓ TRẠNG THÁI: AN TOÀN (Ngưỡng dừng AUTO E-STOP: 1.0 cm)")
            self.safety_desc.setStyleSheet("color: #34D399; font-weight: bold; font-size: 11px;")
            self.safety_badge.setText("SAFETY OK")
            self.safety_badge.setStyleSheet("background: #064E3B; color: #34D399; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 10px;")

    def trigger_estop(self):
        self.is_running = False
        self.vpp_active = False
        self.vpp_auto_armed = False
        self.vpp_state = "STOPPED"
        if getattr(self, "conveyor_running", False):
            self.toggle_conveyor_feeder(False)
        if hasattr(self, "lbl_vpp_phase"):
            self.lbl_vpp_phase.setText("⏹ Đã dừng khẩn cấp (E-STOP)")
            self.lbl_vpp_phase.setStyleSheet("color: #EF4444; font-weight: bold; font-size: 11px;")
        if hasattr(self, "btn_vpp_start"):
            self.btn_vpp_start.setEnabled(True)
            self.btn_vpp_stop.setEnabled(False)
        self.btn_start.setText("START PROGRAM")
        self.btn_start.setStyleSheet("background: #059669; color: white; font-weight: bold;")
        QtWidgets.QMessageBox.warning(self, "E-STOP Triggered", "🚨 Dừng khẩn cấp E-STOP! Toàn bộ tiến trình robot đã bị ngắt an toàn.")

    # ========================================================
    # PROGRAM BUILDER OPERATIONS (BEAUTIFUL CARDS)
    # ========================================================
    def load_program_from_file(self, name):
        filepath = os.path.join(PROGRAMS_DIR, f"{name}.dsr")
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    self.program_steps = json.load(f)
                    self.current_program_name = name
                    return True
            except Exception as e:
                print(f"Error loading program: {e}")
        return False

    def save_program_dialog(self):
        name = self.edit_prog_name.text().strip() or "Custom_Task"
        self.current_program_name = name
        filepath = os.path.join(PROGRAMS_DIR, f"{name}.dsr")
        try:
            with open(filepath, 'w') as f:
                json.dump(self.program_steps, f, indent=2)
            QtWidgets.QMessageBox.information(self, "Lưu thành công", f"✓ Đã lưu chương trình [{name}.dsr] thành công!")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Lỗi", f"Không thể lưu file: {e}")

    def open_load_dialog(self):
        files = [f.replace('.dsr', '') for f in os.listdir(PROGRAMS_DIR) if f.endswith('.dsr')]
        if not files:
            QtWidgets.QMessageBox.information(self, "Thông báo", "Chưa có file chương trình nào được lưu.")
            return

        item, ok = QtWidgets.QInputDialog.getItem(self, "Mở Program Đã Lưu", "Chọn file chương trình:", files, 0, False)
        if ok and item:
            if self.load_program_from_file(item):
                self.edit_prog_name.setText(item)
                self.lbl_prog_name.setText(f"Program: {item}")
                self.refresh_tree_cards()
                self.update_tracker_display()
                QtWidgets.QMessageBox.information(self, "Thành công", f"✓ Đã nạp chương trình [{item}]!")

    def refresh_tree_cards(self):
        """Renders beautiful card widgets matching HTML mockup"""
        # Clear existing cards
        while self.tree_container_layout.count() > 1:
            item = self.tree_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for idx, step in enumerate(self.program_steps):
            card = InstructionCardWidget(
                idx, step,
                on_up=self.move_step_up,
                on_down=self.move_step_down,
                on_delete=self.delete_step,
                on_edit=self.edit_step
            )
            self.tree_container_layout.insertWidget(idx, card)

    def edit_step(self, idx):
        if not (0 <= idx < len(self.program_steps)):
            return
        step = self.program_steps[idx]
        t = step.get('type', '')

        if t == 'Gripper':
            nums = [float(s) for s in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", str(step.get('params', '40')))]
            cur_w = nums[0] if len(nums) > 0 else 40.0
            cur_f = nums[1] if len(nums) > 1 else self.gripper_force
            dlg = GripperConfigDialog(current_width=cur_w, current_force=cur_f, target_name=step.get('target', 'Clamp_Box'), parent=self)
            if dlg.exec_() == QtWidgets.QDialog.Accepted:
                name, params, desc, w, f = dlg.get_result()
                step['target'] = name
                step['params'] = params
                step['desc'] = desc
                self.refresh_tree_cards()
                self.update_tracker_display()
        elif t == 'Wait':
            nums = [float(s) for s in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", str(step.get('params', '1.0')))]
            cur_d = nums[0] if len(nums) > 0 else 1.0
            dlg = DelayConfigDialog(current_delay=cur_d, parent=self)
            if dlg.exec_() == QtWidgets.QDialog.Accepted:
                target, params, desc, secs = dlg.get_result()
                step['target'] = target
                step['params'] = params
                step['desc'] = desc
                self.refresh_tree_cards()
                self.update_tracker_display()
        elif t == 'MoveNMPC':
            text, ok = QtWidgets.QInputDialog.getText(
                self, "Chỉnh sửa MoveNMPC", "Nhập tọa độ mục tiêu [X, Y, Z] (m):",
                text=str(step.get('params', ''))
            )
            if ok and text.strip():
                step['params'] = text.strip()
                self.refresh_tree_cards()
                self.update_tracker_display()
        elif t == 'RotateCircle':
            p_str = str(step.get('params', ''))
            cur_r = 0.14
            cur_s = 60.0
            cur_turns = 1.0
            cur_cx = getattr(self.radar, 'obs_x', 0.45)
            cur_cy = getattr(self.radar, 'obs_y', 0.0)

            m_c = re.search(r"center:\s*\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]", p_str)
            if m_c:
                cur_cx = float(m_c.group(1))
                cur_cy = float(m_c.group(2))

            m_r = re.search(r"radius:\s*([-+]?\d*\.?\d+)", p_str)
            if m_r:
                cur_r = float(m_r.group(1))

            m_s = re.search(r"speed:\s*([-+]?\d*\.?\d+)", p_str)
            if m_s:
                cur_s = float(m_s.group(1))

            m_t = re.search(r"turns:\s*([-+]?\d*\.?\d+)", p_str)
            if m_t:
                cur_turns = float(m_t.group(1))
            else:
                m_a = re.search(r"angle:\s*([-+]?\d*\.?\d+)", p_str)
                if m_a:
                    cur_turns = float(m_a.group(1)) / 360.0

            dlg = OrbitConfigDialog(
                current_radius=cur_r,
                current_speed=cur_s,
                current_turns=cur_turns,
                center_x=cur_cx,
                center_y=cur_cy,
                target_name=step.get('target', 'Full_360_Orbit'),
                parent=self
            )
            if dlg.exec_() == QtWidgets.QDialog.Accepted:
                name, params, desc, r, s, turns, center = dlg.get_config()
                step['target'] = name
                step['params'] = params
                step['desc'] = desc
                self.refresh_tree_cards()
                self.update_tracker_display()
        elif t == 'MoveJ':
            text, ok = QtWidgets.QInputDialog.getText(
                self, "Chỉnh sửa MoveJ", "Nhập góc 6 khớp [J1, J2, J3, J4, J5, J6] (rad):",
                text=str(step.get('params', ''))
            )
            if ok and text.strip():
                step['params'] = text.strip()
                self.refresh_tree_cards()
                self.update_tracker_display()

    def move_step_up(self, idx):
        if idx > 0:
            step = self.program_steps.pop(idx)
            self.program_steps.insert(idx - 1, step)
            self.refresh_tree_cards()
            self.update_tracker_display()

    def move_step_down(self, idx):
        if idx < len(self.program_steps) - 1:
            step = self.program_steps.pop(idx)
            self.program_steps.insert(idx + 1, step)
            self.refresh_tree_cards()
            self.update_tracker_display()

    def delete_step(self, idx):
        if 0 <= idx < len(self.program_steps):
            self.program_steps.pop(idx)
            self.refresh_tree_cards()
            self.update_tracker_display()

    def clear_program_tree(self):
        if QtWidgets.QMessageBox.question(self, "Xác nhận", "Xóa toàn bộ câu lệnh?") == QtWidgets.QMessageBox.Yes:
            self.program_steps.clear()
            self.refresh_tree_cards()
            self.update_tracker_display()

    def new_program(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Tạo Program Mới", "Nhập tên chương trình:")
        if ok and name.strip():
            self.current_program_name = name.strip()
            self.edit_prog_name.setText(self.current_program_name)
            self.program_steps = [
                {"id": 1, "type": "MoveJ", "target": "HOME_SAFE", "params": "[0.0, 0.0, 1.5708, 0.0, 1.5708, 0.0]", "desc": "Khởi đầu an toàn"}
            ]
            self.refresh_tree_cards()
            self.update_tracker_display()

    def add_command_to_tree(self, cmd_type):
        new_id = len(self.program_steps) + 1
        if cmd_type == 'MoveNMPC':
            step = {"id": new_id, "type": "MoveNMPC", "target": f"Point_{new_id}", "params": f"[{self.tcp_pos[0]:.2f}, {self.tcp_pos[1]:.2f}, 0.15]", "desc": "NMPC bẻ cong đường né cản"}
        elif cmd_type == 'RotateCircle':
            obs_x = getattr(self.radar, 'obs_x', 0.45)
            obs_y = getattr(self.radar, 'obs_y', 0.0)
            dlg = OrbitConfigDialog(
                current_radius=0.14,
                current_speed=60.0,
                current_turns=1.0,
                center_x=obs_x,
                center_y=obs_y,
                target_name="Full_360_Orbit",
                parent=self
            )
            if dlg.exec_() == QtWidgets.QDialog.Accepted:
                name, params, desc, r, s, turns, center = dlg.get_config()
                step = {"id": new_id, "type": "RotateCircle", "target": name, "params": params, "desc": desc}
            else:
                return
        elif cmd_type == 'MoveJ':
            step = {"id": new_id, "type": "MoveJ", "target": "HOME_SAFE", "params": "[0.0, 0.0, 1.5708, 0.0, 1.5708, 0.0]", "desc": "Quay khớp về vị trí an toàn (Vuông góc)"}
        elif cmd_type == 'Gripper':
            dlg = GripperConfigDialog(current_width=40.0, current_force=self.gripper_force, target_name="Clamp_Box", parent=self)
            if dlg.exec_() == QtWidgets.QDialog.Accepted:
                name, params, desc, w, f = dlg.get_result()
                step = {"id": new_id, "type": "Gripper", "target": name, "params": params, "desc": desc}
            else:
                return
        elif cmd_type == 'Wait':
            dlg = DelayConfigDialog(current_delay=1.0, parent=self)
            if dlg.exec_() == QtWidgets.QDialog.Accepted:
                target, params, desc, secs = dlg.get_result()
                step = {"id": new_id, "type": "Wait", "target": target, "params": params, "desc": desc}
            else:
                return
        
        self.program_steps.append(step)
        self.refresh_tree_cards()
        self.update_tracker_display()

    def record_current_tcp_to_tree(self):
        new_id = len(self.program_steps) + 1
        step = {
            "id": new_id,
            "type": "MoveNMPC",
            "target": f"Recorded_Pt_{new_id}",
            "params": f"[{self.tcp_pos[0]:.3f}, {self.tcp_pos[1]:.3f}, {self.tcp_pos[2]:.3f}]",
            "desc": "Tọa độ lưu trực tiếp từ Jog"
        }
        self.program_steps.append(step)
        self.refresh_tree_cards()
        self.update_tracker_display()
        QtWidgets.QMessageBox.information(self, "Lưu điểm", f"✓ Đã chốt lưu tọa độ [{self.tcp_pos[0]:.3f}, {self.tcp_pos[1]:.3f}, {self.tcp_pos[2]:.3f}] vào cây lệnh!")

    def record_and_switch_builder(self):
        self.record_current_tcp_to_tree()
        self.tabs.setCurrentWidget(self.tab_builder)

    def apply_to_manager(self):
        name = self.edit_prog_name.text().strip() or "Custom_Task"
        self.current_program_name = name
        self.lbl_prog_name.setText(f"Program: {name}")
        self.update_tracker_display()
        self.tabs.setCurrentWidget(self.tab_manager)

    def update_tracker_display(self):
        while self.tracker_layout.count():
            item = self.tracker_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for idx, step in enumerate(self.program_steps):
            f = QtWidgets.QFrame()
            f.setObjectName("stepCard")
            is_active = (idx == self.current_step_idx and self.is_running)
            is_done = (idx < self.current_step_idx)
            
            if is_active:
                f.setStyleSheet("background: #064E3B; border: 2px solid #34D399; border-radius: 6px; padding: 4px;")
            elif is_done:
                f.setStyleSheet("background: #1E293B; border: 1px solid #475569; border-radius: 6px; padding: 4px;")
            else:
                f.setStyleSheet("background: #0F172A; border: 1px solid #1E293B; border-radius: 6px; padding: 4px;")

            l = QtWidgets.QVBoxLayout(f)
            l.setContentsMargins(6, 4, 6, 4)
            st_lbl = QtWidgets.QLabel(f"{idx+1}. {step['type']}")
            st_lbl.setStyleSheet("font-weight: bold; font-size: 10px; color: #FFFFFF;")
            tg_lbl = QtWidgets.QLabel(step['target'])
            tg_lbl.setStyleSheet("font-size: 9px; color: #94A3B8;")
            l.addWidget(st_lbl)
            l.addWidget(tg_lbl)
            self.tracker_layout.addWidget(f)

    # ========================================================
    # EXECUTION RUNNER & STEP SEQUENCER
    # ========================================================
    def start_program(self):
        """Immediately starts executing from current/first step without skipping"""
        if len(self.program_steps) == 0:
            QtWidgets.QMessageBox.warning(self, "Chưa có lệnh", "Chương trình chưa có bước lệnh nào!")
            return
        self.is_running = True
        self.is_paused = False
        self.btn_pause.setText("⏸ PAUSE")
        self.btn_pause.setStyleSheet("padding: 7px 12px; font-weight: bold; background: #D97706; color: white;")
        self.execute_current_step()
        self.update_tracker_display()

    def pause_program(self):
        if not self.is_running:
            return
        self.is_paused = not self.is_paused
        self.btn_pause.setText("▶ RESUME" if self.is_paused else "⏸ PAUSE")
        self.btn_pause.setStyleSheet(
            "padding: 7px 12px; font-weight: bold; background: #059669; color: white;" if self.is_paused
            else "padding: 7px 12px; font-weight: bold; background: #D97706; color: white;"
        )

    def toggle_auto_manual_mode(self):
        self.is_auto_mode = not self.is_auto_mode
        if self.is_auto_mode:
            self.btn_mode.setText("CHẾ ĐỘ: AUTO")
            self.btn_mode.setStyleSheet("padding: 7px 12px; font-weight: bold; background: #2563EB; color: white;")
        else:
            self.btn_mode.setText("CHẾ ĐỘ: MANUAL")
            self.btn_mode.setStyleSheet("padding: 7px 12px; font-weight: bold; background: #D97706; color: white;")

    def check_go_next_step(self):
        if len(self.program_steps) == 0:
            return
        self.current_step_idx = (self.current_step_idx + 1) % len(self.program_steps)
        self.execute_current_step()
        self.update_tracker_display()

    def check_back_prev_step(self):
        if len(self.program_steps) == 0:
            return
        self.current_step_idx = (self.current_step_idx - 1) % len(self.program_steps)
        self.execute_current_step()
        self.update_tracker_display()

    def reset_program(self):
        self.is_running = False
        self.is_paused = False
        self.step_in_progress = False
        self.current_step_idx = 0
        self.cycle_time = 0.0
        self.lbl_timer.setText("Timer: 00:00.0s")
        self.lbl_prog_step.setText("Step: Sẵn sàng")
        self.btn_pause.setText("⏸ PAUSE")
        self.btn_pause.setStyleSheet("padding: 7px 12px; font-weight: bold; background: #D97706; color: white;")
        self.update_tracker_display()

    @staticmethod
    def _build_workpiece_sdf(model_name: str, kind: str, x: float, y: float, z_world: float) -> str:
        """Build standalone SDF XML for an extra 6cm orange cube or green cylinder spawned at conveyor infeed."""
        if kind == "green_cyl":
            geom = "<cylinder><radius>0.030</radius><length>0.060</length></cylinder>"
            inert = "<ixx>0.000063</ixx><ixy>0</ixy><ixz>0</ixz><iyy>0.000063</iyy><iyz>0</iyz><izz>0.000054</izz>"
            mat = "<ambient>0.05 0.65 0.22 1</ambient><diffuse>0.08 0.85 0.30 1</diffuse>"
            link_name = "cyl_link"
        else:
            geom = "<box><size>0.06 0.06 0.06</size></box>"
            inert = "<ixx>0.000072</ixx><ixy>0</ixy><ixz>0</ixz><iyy>0.000072</iyy><iyz>0</iyz><izz>0.000072</izz>"
            mat = "<ambient>0.92 0.18 0.01 1</ambient><diffuse>1.0 0.16 0.01 1</diffuse>"
            link_name = "box_link"
        return (
            f'<?xml version="1.0"?><sdf version="1.8"><model name="{model_name}">'
            f"<pose>{x:.4f} {y:.4f} {z_world:.4f} 0 0 0</pose>"
            f'<link name="{link_name}">'
            f"<inertial><mass>0.12</mass><inertia>{inert}</inertia></inertial>"
            f'<collision name="collision"><geometry>{geom}</geometry>'
            "<surface><friction><ode><mu>1.5</mu><mu2>1.5</mu2></ode></friction>"
            "<bounce><restitution_coefficient>0.0</restitution_coefficient></bounce>"
            "<contact><ode><kp>1000000.0</kp><kd>100.0</kd><max_vel>0.01</max_vel><min_depth>0.001</min_depth></ode></contact>"
            "</surface></collision>"
            f'<visual name="visual"><geometry>{geom}</geometry><material>{mat}</material></visual>'
            "</link></model></sdf>"
        )

    def _update_spawn_count_ui(self):
        n_extra = len(getattr(self, "spawned_workpiece_names", []))
        if hasattr(self, "lbl_vpp_spawn_count"):
            self.lbl_vpp_spawn_count.setText(f"📦 Phôi trên băng tải: 2 gốc (+{n_extra} thêm)")

    def spawn_additional_workpiece(self, kind=None):
        """Spawn an extra orange cube or green cylinder at the conveyor far end (Z >= 2x height clear)."""
        if kind is None:
            idx = self.cmb_spawn_shape.currentIndex() if hasattr(self, "cmb_spawn_shape") else 0
            kind = "green_cyl" if idx == 1 else "orange_cube"
        self._spawn_counter += 1
        model_name = f"spawned_{kind}_{self._spawn_counter}"
        # Far infeed end of conveyor (Y: -0.95..-0.30) and elevated Z=0.905m (bottom 12.5cm > 2x6cm above belt)
        lane_x = (0.45, 0.39, 0.51)[(self._spawn_counter - 1) % 3]
        spawn_y = -0.85
        spawn_z = 0.905
        sdf_xml = self._build_workpiece_sdf(model_name, kind, lane_x, spawn_y, spawn_z)
        self.spawned_workpiece_names.append(model_name)
        self._update_spawn_count_ui()

        wname = "m0609_dynamic_obstacle_world" if self.sim_scenario == "dynamic" else "m0609_rail_world"
        if entity_factory_pb2 is not None and boolean_pb2 is not None and HAVE_GZ_TRANSPORT:
            def _do_create():
                try:
                    n = self.gz_node or gz_transport.Node()
                    req = entity_factory_pb2.EntityFactory()
                    req.sdf = sdf_xml
                    req.name = model_name
                    n.request(f"/world/{wname}/create", req, entity_factory_pb2.EntityFactory, boolean_pb2.Boolean, 1500)
                except Exception:
                    pass
            threading.Thread(target=_do_create, daemon=True).start()

        tag = "Trụ Xanh" if kind == "green_cyl" else "Vuông Cam"
        if hasattr(self, "lbl_vpp_phase") and not self.vpp_active:
            self._vpp_set_phase_ui(
                f"➕ Đã thêm {tag} tại đầu kia băng tải [{lane_x:.2f}, {spawn_y:.2f}, Z={spawn_z:.3f}m].",
                0, "#38BDF8"
            )
        return model_name

    def reset_target_box_pose(self):
        """Remove all spawned extra workpieces, restore ONLY the 2 initial workpieces + obstacle, and reset 2x2 counters."""
        if getattr(self, "conveyor_running", False):
            self.toggle_conveyor_feeder(False)
        else:
            self._publish_conveyor_velocity(0.0)

        wname = "m0609_dynamic_obstacle_world" if self.sim_scenario == "dynamic" else "m0609_rail_world"
        to_remove = list(getattr(self, "spawned_workpiece_names", []))
        self.spawned_workpiece_names.clear()
        self._spawn_counter = 0
        self._update_spawn_count_ui()

        if to_remove and entity_pb2 is not None and boolean_pb2 is not None and HAVE_GZ_TRANSPORT:
            def _do_remove():
                try:
                    n = self.gz_node or gz_transport.Node()
                    for mname in to_remove:
                        rem = entity_pb2.Entity()
                        rem.name = str(mname)
                        rem.type = entity_pb2.Entity.MODEL
                        n.request(f"/world/{wname}/remove", rem, entity_pb2.Entity, boolean_pb2.Boolean, 1000)
                except Exception:
                    pass
            threading.Thread(target=_do_remove, daemon=True).start()

        reqs = [
            f"gz service -s /world/{wname}/set_pose --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 1500 --req 'name: \"target_orange_cube\", position: {{x: 0.45, y: -0.38, z: 0.781}}, orientation: {{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}'",
            f"gz service -s /world/{wname}/set_pose --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 1500 --req 'name: \"target_green_cyl\", position: {{x: 0.45, y: -0.58, z: 0.781}}, orientation: {{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}'",
        ]
        if self.sim_scenario != "dynamic":
            reqs.append(
                f"gz service -s /world/{wname}/set_pose --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 1500 --req 'name: \"obstacle_rail_yellow\", position: {{x: 0.50, y: 0.0, z: 0.894}}, orientation: {{x: 0.0, y: 0.0, z: 0.0, w: 1.0}}'"
            )
        cmd = self._build_ros_shell_cmd(" ; ".join(reqs))
        subprocess.Popen(cmd, shell=True)
        self.vpp_auto_orange_idx = 0
        self.vpp_auto_green_idx = 0
        self.vpp_stable_pose = None
        if hasattr(self, "lbl_vpp_tray_counts"):
            self._update_vpp_tray_counts_ui()
        if hasattr(self, "lbl_vpp_phase") and not self.vpp_active:
            self._vpp_set_phase_ui(
                f"🔄 Đã Reset (xóa {len(to_remove)} phôi thêm, chỉ giữ 2 phôi gốc & làm mới Ô 2×2).",
                0, "#34D399"
            )

    def execute_current_step(self):
        if len(self.program_steps) == 0:
            return
        step = self.program_steps[self.current_step_idx]
        self.step_in_progress = True
        self.step_timer = 0.0
        self.settle_timer = 0.0
        self.orbit_phase_angle = 0.0

        t = step['type']
        if t == 'MoveNMPC':
            try:
                coords = json.loads(step['params'])
                self.current_target_xyz = [float(coords[0]), float(coords[1]), float(coords[2])]
                yaw = float(coords[3]) if len(coords) >= 4 else None
                self.ros.publish_target(
                    self.current_target_xyz[0], self.current_target_xyz[1], self.current_target_xyz[2],
                    yaw=yaw
                )
                self.radar.update_tcp(self.tcp_pos[0], self.tcp_pos[1], self.tcp_pos[2])
            except Exception as e:
                print(f"Error parsing MoveNMPC: {e}")
        elif t == 'MoveJ':
            try:
                joints = json.loads(step['params'])
                self.send_joint_pose(joints)
            except Exception as e:
                print(f"Error parsing MoveJ: {e}")
        elif t == 'Gripper':
            try:
                p_str = str(step['params'])
                nums = [float(s) for s in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", p_str)]
                w = nums[0] if len(nums) > 0 else 40.0
                if len(nums) > 1:
                    f = nums[1]
                    self.force_slider.setValue(int(f))
                    self.on_force_slider(f)
                self.grip_slider.setValue(int(w))
                self.on_gripper_slider(w)
            except Exception as e:
                print(f"Error parsing Gripper: {e}")
        elif t == 'Wait':
            try:
                p_str = str(step['params'])
                nums = [float(s) for s in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", p_str)]
                self.wait_duration = nums[0] if len(nums) > 0 else 1.0
            except Exception:
                self.wait_duration = 1.0
        elif t == 'RotateCircle':
            p_str = str(step.get('params', ''))
            cx = getattr(self.radar, 'obs_x', 0.45)
            cy = getattr(self.radar, 'obs_y', 0.0)
            cz = self.tcp_pos[2]
            radius = 0.14
            speed = 60.0
            total_angle = 360.0

            m_c = re.search(r"center:\s*\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]", p_str)
            if m_c:
                cx = float(m_c.group(1))
                cy = float(m_c.group(2))

            m_r = re.search(r"radius:\s*([-+]?\d*\.?\d+)", p_str)
            if m_r:
                radius = float(m_r.group(1))

            m_s = re.search(r"speed:\s*([-+]?\d*\.?\d+)", p_str)
            if m_s:
                speed = float(m_s.group(1))

            m_t = re.search(r"turns:\s*([-+]?\d*\.?\d+)", p_str)
            if m_t:
                total_angle = float(m_t.group(1)) * 360.0
            else:
                m_a = re.search(r"angle:\s*([-+]?\d*\.?\d+)", p_str)
                if m_a:
                    total_angle = float(m_a.group(1))

            self.orbit_center = [cx, cy, cz]
            self.orbit_radius = radius
            self.orbit_speed = speed
            self.orbit_target_angle = total_angle

            # Continuous angle tracking: calculate starting angle relative to orbit center
            dx = self.tcp_pos[0] - cx
            dy = self.tcp_pos[1] - cy
            dist = math.hypot(dx, dy)
            if dist > 0.01:
                self.orbit_start_angle = math.degrees(math.atan2(dy, dx))
            else:
                self.orbit_start_angle = 0.0

            self.orbit_traveled_angle = 0.0
            self.orbit_phase_angle = 0.0

    def on_force_slider(self, val):
        self.gripper_force = float(val)
        self.lbl_force_val.setText(f"Lực kẹp mục tiêu: {val} N (Bảo vệ phôi, chống bẹp)")
        if hasattr(self, "force_slider_tab6") and self.force_slider_tab6.value() != int(val):
            self.force_slider_tab6.blockSignals(True)
            self.force_slider_tab6.setValue(int(val))
            self.force_slider_tab6.blockSignals(False)
        if hasattr(self, "lbl_force_val_tab6"):
            self.lbl_force_val_tab6.setText(f"💪 Giới hạn lực: {int(val)} N")

    # ========================================================
    # NMPC PARAMETERS & PROCESS LAUNCHER
    # ========================================================
    def apply_nmpc_params(self):
        ox = self.sp_obs_x.value()
        oy = self.sp_obs_y.value()
        oz = self.sp_obs_z.value()
        r = self.sp_obs_r.value() / 100.0
        warn = self.sp_warn_m.value() / 100.0
        danger = self.sp_danger_m.value() / 100.0

        self.danger_margin = danger
        self.warn_margin = warn
        self.radar.update_obstacle(ox, oy, r, warn, danger)
        self.ros.publish_obstacle(ox, oy, oz, z_top=max(0.25, oz * 1.67), radius=r)

        QtWidgets.QMessageBox.information(
            self, "Áp dụng thành công",
            f"✓ Đã cập nhật thông số NMPC & Vật cản!\n• Tọa độ: [{ox:.2f}, {oy:.2f}, {oz:.2f}]m\n• Bán kính: {r*100:.0f}cm | Đệm cảnh báo: {warn*100:.0f}cm\n• Ngưỡng ngắt AUTO E-STOP: {danger*100:.1f}cm\n• Radar 2D và bộ giải NMPC đã cập nhật."
        )
        self.tabs.setCurrentWidget(self.tab_manager)

    # ========================================================
    # SCENARIO SWITCHER (RAIL DEFAULT vs DYNAMIC OBSTACLE)
    # ========================================================
    def _refresh_scenario_styles(self):
        is_dyn = (self.sim_scenario == "dynamic")
        rail_slide = hasattr(self, "chk_rail_slide") and self.chk_rail_slide.isChecked()
        self.btn_scen_static.blockSignals(True)
        self.btn_scen_dynamic.blockSignals(True)
        self.btn_scen_static.setChecked(not is_dyn)
        self.btn_scen_dynamic.setChecked(is_dyn)
        self.btn_scen_static.blockSignals(False)
        self.btn_scen_dynamic.blockSignals(False)

        self.btn_scen_static.setStyleSheet(
            "background: #0284C7; color: white; font-weight: bold; padding: 8px 14px; border-radius: 6px;"
            if not is_dyn else "background: #1E293B; color: #94A3B8; padding: 8px 14px; border-radius: 6px;"
        )
        self.btn_scen_dynamic.setStyleSheet(
            "background: #D97706; color: white; font-weight: bold; padding: 8px 14px; border-radius: 6px;"
            if is_dyn else "background: #1E293B; color: #94A3B8; padding: 8px 14px; border-radius: 6px;"
        )

        self.btn_dyn_circle.setEnabled(is_dyn)
        self.btn_dyn_zigzag.setEnabled(is_dyn)
        if hasattr(self, "chk_rail_slide"):
            self.chk_rail_slide.setEnabled(not is_dyn)
        self.btn_dyn_pause_toggle.setEnabled(is_dyn or rail_slide)
        self.sl_dyn_speed.setEnabled(is_dyn or rail_slide)

        self.btn_dyn_circle.blockSignals(True)
        self.btn_dyn_zigzag.blockSignals(True)
        self.btn_dyn_circle.setChecked(self.dyn_obs_mode == "circle")
        self.btn_dyn_zigzag.setChecked(self.dyn_obs_mode == "zigzag")
        self.btn_dyn_circle.blockSignals(False)
        self.btn_dyn_zigzag.blockSignals(False)

        self.btn_dyn_pause_toggle.blockSignals(True)
        self.btn_dyn_pause_toggle.setChecked(self.dyn_obs_paused)
        self.btn_dyn_pause_toggle.blockSignals(False)

        if is_dyn or rail_slide:
            if self.dyn_obs_paused:
                self.btn_dyn_pause_toggle.setText("⏸ ĐANG ĐỨNG YÊN (Bấm để Tiếp Tục Di Chuyển)")
                self.btn_dyn_pause_toggle.setStyleSheet(
                    "background: #B91C1C; color: white; font-weight: bold; padding: 8px 12px; border-radius: 6px;"
                )
            else:
                self.btn_dyn_pause_toggle.setText("▶ ĐANG DI CHUYỂN (Bấm để Dừng Đứng Yên)")
                self.btn_dyn_pause_toggle.setStyleSheet(
                    "background: #059669; color: white; font-weight: bold; padding: 8px 12px; border-radius: 6px;"
                )

        if is_dyn:
            self.btn_dyn_circle.setStyleSheet(
                "background: #EA580C; color: white; font-weight: bold; padding: 7px 12px; border-radius: 6px;"
                if self.dyn_obs_mode == "circle" else "background: #1E293B; color: #CBD5E1; padding: 7px 12px;"
            )
            self.btn_dyn_zigzag.setStyleSheet(
                "background: #DB2777; color: white; font-weight: bold; padding: 7px 12px; border-radius: 6px;"
                if self.dyn_obs_mode == "zigzag" else "background: #1E293B; color: #CBD5E1; padding: 7px 12px;"
            )
            mode_tag = "Hình Tròn" if self.dyn_obs_mode == "circle" else "Zigzag Khó Đoán"
            self.lbl_scen_world_badge.setText(f"World: pick_and_place_dynamic.sdf ({mode_tag} vùng 30cm)")
            self.lbl_scen_world_badge.setStyleSheet("color: #FDBA74; font-family: monospace; font-size: 11px; font-weight: bold;")
            if hasattr(self, "lbl_gz_world_desc"):
                self.lbl_gz_world_desc.setText("1. Gazebo Harmonic Sim (pick_and_place_dynamic.sdf + Vật Động Tròn/Zigzag)")
        else:
            for b in (self.btn_dyn_circle, self.btn_dyn_zigzag):
                b.setStyleSheet("background: #0F172A; color: #475569; padding: 7px 12px;")
            if not rail_slide:
                self.btn_dyn_pause_toggle.setStyleSheet("background: #0F172A; color: #475569; padding: 7px 12px;")
                self.btn_dyn_pause_toggle.setText("⏸ Đang đứng yên tại ray HOME (Bật checkbox trên để trượt)")
            self.lbl_scen_world_badge.setText("World: pick_and_place_rail.sdf (Thanh ray tại HOME — Mặc định)")
            self.lbl_scen_world_badge.setStyleSheet("color: #38BDF8; font-family: monospace; font-size: 11px; font-weight: bold;")
            if hasattr(self, "lbl_gz_world_desc"):
                self.lbl_gz_world_desc.setText("1. Gazebo Harmonic Sim (pick_and_place_rail.sdf + ros2_control)")
            if hasattr(self, "lbl_dyn_live_state") and not rail_slide:
                self.lbl_dyn_live_state.setText("Trạng thái: Đang ở Bản 1 — Thanh ray tại HOME (Trụ Vàng Ø10×26cm)")
                self.lbl_dyn_live_vel.setText("Vận tốc (Vx, Vy): [0.00, 0.00] m/s | Độ lớn |V|: 0.00 m/s")

    def on_toggle_rail_slide(self, checked):
        if self.sim_scenario != "rail":
            return
        self._refresh_scenario_styles()
        if checked:
            self.on_toggle_vision_obs_sync(True)
            self.dyn_obs_mode = "rail"
            dyn_proc = self.processes.get('dynamic_obs')
            if dyn_proc is None or dyn_proc.poll() is not None:
                self.start_dynamic_obstacle_process()
            self.ros.publish_dyn_mode("rail")
            self.ros.publish_dyn_speed(self.dyn_obs_speed)
            self.ros.publish_dyn_paused(self.dyn_obs_paused)
        else:
            self.stop_dynamic_obstacle_process()
            self.radar.update_obstacle(0.50, 0.0, 0.05, self.warn_margin, self.danger_margin, is_dynamic=False)
            self.ros.publish_obstacle(0.50, 0.0, 0.13, z_top=0.274, radius=0.05)

    def on_select_scenario(self, scenario: str, auto_launch_gz: bool = True):
        if scenario == "static":
            scenario = "rail"
        if scenario not in ("rail", "dynamic"):
            return
        prev = self.sim_scenario
        self.sim_scenario = scenario
        self.radar.clear_obs_trail()

        target_world = "pick_and_place_dynamic.sdf" if scenario == "dynamic" else "pick_and_place_rail.sdf"
        if scenario == "rail":
            self.dyn_obs_mode = "circle"
            if hasattr(self, "chk_rail_slide"):
                self.chk_rail_slide.blockSignals(True)
                self.chk_rail_slide.setChecked(False)
                self.chk_rail_slide.blockSignals(False)
            self.stop_dynamic_obstacle_process()
            self.radar.update_obstacle(0.50, 0.0, 0.05, self.warn_margin, self.danger_margin, is_dynamic=False)
            self.ros.publish_obstacle(0.50, 0.0, 0.13, z_top=0.274, radius=0.05)
        else:
            if self.dyn_obs_mode not in ("circle", "zigzag"):
                self.dyn_obs_mode = "circle"
            self.on_toggle_vision_obs_sync(True)
        self._refresh_scenario_styles()

        if auto_launch_gz:
            gz_proc = self.processes.get('gazebo')
            gz_alive = gz_proc is not None and gz_proc.poll() is None
            if (not gz_alive) or (self.running_gz_world != target_world) or (prev != scenario):
                self.start_gazebo_process(show_popup=False)

    def on_select_dyn_mode(self, mode: str):
        if mode not in ("circle", "zigzag", "rail"):
            return
        self.dyn_obs_mode = mode
        self._refresh_scenario_styles()
        self.radar.clear_obs_trail()
        if self.sim_scenario == "dynamic" or mode == "rail":
            self.on_toggle_vision_obs_sync(True)
            dyn_proc = self.processes.get('dynamic_obs')
            if dyn_proc is None or dyn_proc.poll() is not None:
                self.start_dynamic_obstacle_process()
            self.ros.publish_dyn_mode(mode)
            self.ros.publish_dyn_speed(self.dyn_obs_speed)

    def toggle_dyn_obstacle_motion(self):
        self.dyn_obs_paused = not self.dyn_obs_paused
        self._refresh_scenario_styles()
        if self.sim_scenario == "dynamic" or (hasattr(self, "chk_rail_slide") and self.chk_rail_slide.isChecked()):
            self.ros.publish_dyn_paused(self.dyn_obs_paused)

    def on_dyn_speed_changed(self, val):
        self.dyn_obs_speed = float(val) / 10.0
        self.lbl_dyn_speed.setText(f"{self.dyn_obs_speed:.1f}x")
        if self.sim_scenario == "dynamic" or (hasattr(self, "chk_rail_slide") and self.chk_rail_slide.isChecked()):
            self.ros.publish_dyn_speed(self.dyn_obs_speed)

    def on_dyn_obs_state(self, data):
        is_rail_slide = (self.sim_scenario == "rail" and hasattr(self, "chk_rail_slide") and self.chk_rail_slide.isChecked())
        if (self.sim_scenario != "dynamic" and not is_rail_slide) or len(data) < 7:
            return
        ox, oy, vx, vy, oz_top, orad, mode_id = data
        if mode_id < 0.5:
            return
        self.vision_obs_info = [ox, oy, oz_top, orad]
        self.vision_obs_ts = time.monotonic()
        safe_r = max(0.05, min(0.20, orad + 0.015))
        self.radar.update_obstacle(ox, oy, safe_r, self.warn_margin, self.danger_margin, vx=vx, vy=vy, is_dynamic=True)

        m_int = int(round(mode_id))
        mode_txt = "TRÒN" if m_int == 1 else ("ZIGZAG" if m_int == 2 else "RAY TRƯỢT")
        vmag = math.hypot(vx, vy)
        if hasattr(self, "lbl_dyn_live_pos"):
            self.lbl_dyn_live_pos.setText(
                f"Vị trí XY: [{ox:.3f}, {oy:.3f}] m | H = {oz_top*100:.0f} cm | R = {orad*100:.1f} cm"
            )
            self.lbl_dyn_live_vel.setText(
                f"Vận tốc (Vx, Vy): [{vx:+.2f}, {vy:+.2f}] m/s | Độ lớn |V|: {vmag:.2f} m/s"
            )
            state_txt = (
                f"⏸ ĐANG ĐỨNG YÊN ({mode_txt})" if self.dyn_obs_paused or vmag < 0.005
                else f"▶ ĐANG DI CHUYỂN ({mode_txt} | Hệ số {self.dyn_obs_speed:.1f}x)"
            )
            self.lbl_dyn_live_state.setText(f"Trạng thái: {state_txt}")
            self.lbl_dyn_live_state.setStyleSheet(
                "color: #FBBF24; font-weight: bold; font-size: 11px;"
                if self.dyn_obs_paused else "color: #34D399; font-weight: bold; font-size: 11px;"
            )
        if hasattr(self, "lbl_vis_obs_xyz"):
            self.lbl_vis_obs_xyz.setText(f"XY: [{ox:.3f}, {oy:.3f}]m | V:({vx:+.2f},{vy:+.2f})")
            self.lbl_vis_obs_dim.setText(f"ĐỘNG ({mode_txt}) | H:{oz_top*100:.0f}cm R:{orad*100:.1f}cm")
        if hasattr(self, "lbl_vpp_obs_summary"):
            self.lbl_vpp_obs_summary.setText(f"🟡 {mode_txt} [{ox:.2f}, {oy:.2f}] V=({vx:+.2f},{vy:+.2f})")

    @staticmethod
    def _resolve_ros_setup():
        """Find installed ROS 2 setup.bash across distros on the current machine."""
        distro = os.environ.get("ROS_DISTRO", "").strip()
        candidates = []
        if distro:
            candidates.append(f"/opt/ros/{distro}/setup.bash")
        candidates.extend(["/opt/ros/jazzy/setup.bash", "/opt/ros/humble/setup.bash"])
        candidates.extend(sorted(glob.glob("/opt/ros/*/setup.bash")))
        for path in candidates:
            if path and os.path.isfile(path):
                return path
        return None

    @staticmethod
    def _resolve_ws_setup():
        """Locate the active ROS 2 workspace install/setup.bash dynamically."""
        candidates = []
        try:
            from ament_index_python.packages import get_package_prefix
            pkg_prefix = get_package_prefix("dsr_m0609_nmpc")
            candidates.append(os.path.join(os.path.dirname(pkg_prefix), "setup.bash"))
            candidates.append(os.path.join(pkg_prefix, "local_setup.bash"))
        except Exception:
            pass
        for anchor in (os.path.abspath(__file__), os.path.realpath(__file__)):
            p = Path(anchor)
            for parent in p.parents:
                candidates.append(str(parent / "install" / "setup.bash"))
                if parent.name == "install":
                    candidates.append(str(parent / "setup.bash"))
        for prefix in os.environ.get("COLCON_PREFIX_PATH", "").split(":"):
            if prefix:
                candidates.append(os.path.join(prefix, "setup.bash"))
        for path in candidates:
            if path and os.path.isfile(path):
                return path
        return None

    @staticmethod
    def _resolve_acados_lib_dirs():
        """Locate acados shared library directories dynamically."""
        candidates = []
        for env_var in ("ACADOS_SOURCE_DIR", "ACADOS_DIR"):
            val = os.environ.get(env_var, "").strip()
            if val:
                candidates.append(os.path.join(val, "lib"))
        candidates.extend([
            os.path.join(os.path.expanduser("~"), "acados", "lib"),
            "/opt/acados/lib",
            "/usr/local/acados/lib",
        ])
        for anchor in (os.path.abspath(__file__), os.path.realpath(__file__)):
            for parent in Path(anchor).parents:
                candidates.append(str(parent / "acados" / "lib"))
        try:
            from ament_index_python.packages import get_package_prefix
            candidates.append(os.path.join(get_package_prefix("dsr_m0609_nmpc"), "lib"))
        except Exception:
            pass
        return list(dict.fromkeys(d for d in candidates if os.path.isdir(d)))

    def _build_ros_shell_cmd(self, ros_cmd: str, with_acados: bool = False) -> str:
        """Build a portable bash -c command sourcing ROS 2, workspace setup, and acados libs."""
        parts = []
        ros_setup = self._resolve_ros_setup()
        if ros_setup:
            parts.append(f"source {shlex.quote(ros_setup)}")
        ws_setup = self._resolve_ws_setup()
        if ws_setup:
            parts.append(f"source {shlex.quote(ws_setup)}")
        if with_acados:
            lib_dirs = self._resolve_acados_lib_dirs()
            if lib_dirs:
                joined = ":".join(lib_dirs)
                parts.append(f"export LD_LIBRARY_PATH={shlex.quote(joined)}:$LD_LIBRARY_PATH")
        parts.append(ros_cmd)
        inner = " && ".join(parts)
        return f"bash -c {shlex.quote(inner)}"

    def start_dynamic_obstacle_process(self):
        dyn_proc = self.processes.get('dynamic_obs')
        if dyn_proc and dyn_proc.poll() is None:
            self.ros.publish_dyn_mode(self.dyn_obs_mode)
            self.ros.publish_dyn_speed(self.dyn_obs_speed)
            self.ros.publish_dyn_paused(self.dyn_obs_paused)
            return
        try:
            subprocess.run("pkill -9 -f 'dynamic_obstacle_node'", shell=True, timeout=2)
            time.sleep(0.2)
        except Exception:
            pass
        cmd = self._build_ros_shell_cmd(
            f"ros2 run dsr_m0609_nmpc dynamic_obstacle_node.py --mode {self.dyn_obs_mode}"
        )
        p = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
        self.processes['dynamic_obs'] = p
        QTimer.singleShot(800, lambda: self.ros.publish_dyn_mode(self.dyn_obs_mode))
        QTimer.singleShot(900, lambda: self.ros.publish_dyn_speed(self.dyn_obs_speed))
        QTimer.singleShot(950, lambda: self.ros.publish_dyn_paused(self.dyn_obs_paused))

    def stop_dynamic_obstacle_process(self):
        p = self.processes.pop('dynamic_obs', None)
        if p:
            try:
                os.killpg(os.getpgid(p.pid), 9)
            except Exception:
                pass
        try:
            subprocess.run("pkill -9 -f 'dynamic_obstacle_node'", shell=True, timeout=2)
        except Exception:
            pass

    def start_gazebo_process(self, show_popup=True):
        # Kill previous stale gazebo processes first to prevent duplicate windows!
        self.stop_dynamic_obstacle_process()
        try:
            subprocess.run(
                "pkill -9 -f 'gazebo|gz sim|gz-sim|ros_gz_sim|ros_gz_bridge|parameter_bridge|gz-sim-gui'",
                shell=True, timeout=2
            )
            time.sleep(0.4)
        except Exception:
            pass
        world_name = "pick_and_place_dynamic.sdf" if self.sim_scenario == "dynamic" else "pick_and_place_rail.sdf"
        self.running_gz_world = world_name
        cmd = self._build_ros_shell_cmd(
            f"ros2 launch dsr_m0609_nmpc gazebo_pick_and_place.launch.py world_name:={world_name}"
        )
        p = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
        self.processes['gazebo'] = p
        if self.sim_scenario == "dynamic":
            QTimer.singleShot(2500, self.start_dynamic_obstacle_process)
        if show_popup:
            QtWidgets.QMessageBox.information(self, "Process Started", f"Gazebo Sim ({world_name}) đang khởi động nền.")

    def stop_gazebo_process(self):
        self.stop_dynamic_obstacle_process()
        self.running_gz_world = None
        p = self.processes.pop('gazebo', None)
        if p:
            try:
                os.killpg(os.getpgid(p.pid), 9)
            except Exception:
                pass
        # Complete sweep kill of all Gazebo Sim GUI and server processes!
        try:
            subprocess.run(
                "pkill -9 -f 'gazebo|gz sim|gz-sim|ros_gz_sim|ros_gz_bridge|parameter_bridge|gz-sim-gui'",
                shell=True, timeout=2
            )
        except Exception:
            pass
        self.lbl_gz_status.setText("🔴 ĐÃ DỪNG")
        self.lbl_gz_status.setStyleSheet("color: #EF4444; font-weight: bold; font-family: monospace;")
        self.btn_start_gz.setEnabled(True)
        self.btn_stop_gz.setEnabled(False)

    def start_nmpc_process(self):
        # Kill previous stale nmpc node if any
        try:
            subprocess.run("pkill -9 -f 'm0609_nmpc_node'", shell=True, timeout=2)
            time.sleep(0.3)
        except Exception:
            pass
        cmd = self._build_ros_shell_cmd("ros2 run dsr_m0609_nmpc m0609_nmpc_node", with_acados=True)
        p = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
        self.processes['nmpc'] = p
        QtWidgets.QMessageBox.information(self, "Process Started", "acados NMPC Node đang chạy nền.")

    def stop_nmpc_process(self):
        p = self.processes.pop('nmpc', None)
        if p:
            try:
                os.killpg(os.getpgid(p.pid), 9)
            except Exception:
                pass
        try:
            subprocess.run("pkill -9 -f 'm0609_nmpc_node'", shell=True, timeout=2)
        except Exception:
            pass
        self.lbl_nmpc_status.setText("🔴 ĐÃ DỪNG")
        self.lbl_nmpc_status.setStyleSheet("color: #EF4444; font-weight: bold; font-family: monospace;")
        self.btn_start_nmpc.setEnabled(True)
        self.btn_stop_nmpc.setEnabled(False)

    # ========================================================
    # D435 CAMERA VISION HANDLERS & LAUNCHER
    # ========================================================
    def on_vision_image(self, qimg, is_debug):
        pix = QtGui.QPixmap.fromImage(qimg)
        for lbl in (getattr(self, "lbl_vision_video", None), getattr(self, "lbl_vision_video_tab6", None)):
            if lbl is not None:
                rect = lbl.contentsRect()
                tw = max(120, rect.width() - 2)
                th = max(90, rect.height() - 2)
                scaled = pix.scaled(tw, th, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                lbl.setPixmap(scaled)
        for badge in (getattr(self, "lbl_cam_stream_badge", None), getattr(self, "lbl_cam_stream_badge_tab6", None)):
            if badge is not None:
                if is_debug:
                    badge.setText("🟢 LIVE VISION (D435 30Hz | OpenCV + TF2)")
                    badge.setStyleSheet("color: #10B981; font-weight: bold; font-family: monospace; font-size: 10px;")
                else:
                    badge.setText("🟡 RAW D435 FEED (Bấm ▶ Bật Vision để đo 3D)")
                    badge.setStyleSheet("color: #FBBF24; font-weight: bold; font-family: monospace; font-size: 10px;")

    def on_vision_box(self, data):
        self.vision_box_pose = data
        self.vision_box_ts = time.monotonic()
        bx, by, bz, yaw = data
        self.lbl_vis_box_xyz.setText(f"XYZ: [{bx:.3f}, {by:.3f}, {bz:.3f}] m")
        dims = getattr(self, "vision_box_dims", None)
        if dims and len(dims) >= 3:
            dim_txt = f"Yaw:{math.degrees(yaw):.0f}° | {dims[0]*100:.1f}×{dims[1]*100:.1f}×{dims[2]*100:.1f}cm"
        else:
            dim_txt = f"Yaw (J6): {math.degrees(yaw):.1f}°"
        self.lbl_vis_box_yaw.setText(dim_txt)
        if hasattr(self, "lbl_vpp_box_xyz"):
            self.lbl_vpp_box_xyz.setText(f"Ô gắp: [{bx:.3f}, {by:.3f}, {bz:.3f}] m")
        if getattr(self, "vision_box_state", "OK") != "CARRIED":
            self.lbl_vis_box_status.setText("🟢 Đã quét & lưu chuẩn")
            self.lbl_vis_box_status.setStyleSheet("font-size: 10px; font-weight: bold; color: #34D399;")

    def on_vision_obs(self, data):
        if self.sim_scenario == "dynamic":
            return
        self.vision_obs_info = data
        self.vision_obs_ts = time.monotonic()
        ox, oy, oz_top, orad = data
        self.lbl_vis_obs_xyz.setText(f"XY: [{ox:.3f}, {oy:.3f}]m | R:{orad*100:.1f}cm")
        dims = getattr(self, "vision_obs_dims", None)
        if dims and len(dims) >= 3:
            self.lbl_vis_obs_dim.setText(
                f"W×D×H: {dims[0]*100:.1f}×{dims[1]*100:.1f}×{dims[2]*100:.1f} cm"
            )
        else:
            self.lbl_vis_obs_dim.setText(f"W×D×H: {orad*200:.1f}×{orad*200:.1f}×{oz_top*100:.1f} cm")
        if hasattr(self, "lbl_vpp_obs_summary"):
            self.lbl_vpp_obs_summary.setText(f"🛑 Cản [{ox:.2f}, {oy:.2f}] H={oz_top:.2f}m R={orad*100:.1f}cm")

        if self.vision_sync_obs:
            safe_r = max(0.05, min(0.20, orad + 0.02))
            self.radar.update_obstacle(ox, oy, safe_r, self.warn_margin, self.danger_margin)
            self.ros.publish_obstacle(ox, oy, oz_top * 0.6, z_top=oz_top, radius=orad)

    def on_vision_status(self, st):
        box_state = st.get("box_state", "MISSING")
        obs_state = st.get("obs_state", "MISSING")
        self.vision_box_state = box_state
        if st.get("box_dims"):
            self.vision_box_dims = [float(v) for v in st["box_dims"][:3]]
        if st.get("obs_dims"):
            self.vision_obs_dims = [float(v) for v in st["obs_dims"][:3]]

        target_kind = st.get("target_kind")
        self.vpp_active_target_kind = target_kind
        orange_pose = st.get("orange_pose")
        green_pose = st.get("green_pose")
        self.vpp_orange_pose = [float(v) for v in orange_pose[:4]] if orange_pose else None
        self.vpp_green_pose = [float(v) for v in green_pose[:4]] if green_pose else None
        self.vpp_orange_in_zone = bool(st.get("orange_in_zone", False))
        self.vpp_green_in_zone = bool(st.get("green_in_zone", False))
        self.vpp_conveyor_items = list(st.get("conveyor_items") or [])

        if hasattr(self, "lbl_vis_slot3_xyz"):
            if self.vpp_orange_pose:
                ox, oy, oz, oyaw = self.vpp_orange_pose
                self.lbl_vis_slot3_xyz.setText(f"XYZ: [{ox:.3f}, {oy:.3f}, {oz:.3f}]")
                ztxt = "🟢 TRONG Ô GẮP" if self.vpp_orange_in_zone else "🟡 CHỜ BĂNG TẢI"
                self.lbl_vis_slot3_info.setText(f"{ztxt} | 6×6×6cm")
                self.lbl_vis_slot3_info.setStyleSheet(
                    f"font-size: 10px; color: {'#34D399' if self.vpp_orange_in_zone else '#FBBF24'}; font-weight: bold;"
                )
            else:
                self.lbl_vis_slot3_xyz.setText("XYZ: ---, ---, ---")
                self.lbl_vis_slot3_info.setText("⚪ Không thấy trên băng tải")
                self.lbl_vis_slot3_info.setStyleSheet("font-size: 10px; color: #94A3B8;")

        if hasattr(self, "lbl_vis_slot4_xyz"):
            if self.vpp_green_pose:
                gx, gy, gz, gyaw = self.vpp_green_pose
                self.lbl_vis_slot4_xyz.setText(f"XYZ: [{gx:.3f}, {gy:.3f}, {gz:.3f}]")
                ztxt = "🟢 TRONG Ô GẮP" if self.vpp_green_in_zone else "🟡 CHỜ BĂNG TẢI"
                self.lbl_vis_slot4_info.setText(f"{ztxt} | Ø6×6cm")
                self.lbl_vis_slot4_info.setStyleSheet(
                    f"font-size: 10px; color: {'#34D399' if self.vpp_green_in_zone else '#FBBF24'}; font-weight: bold;"
                )
            else:
                self.lbl_vis_slot4_xyz.setText("XYZ: ---, ---, ---")
                self.lbl_vis_slot4_info.setText("⚪ Không thấy trên băng tải")
                self.lbl_vis_slot4_info.setStyleSheet("font-size: 10px; color: #94A3B8;")

        # Update ALL detected conveyor workpieces simultaneously in Tab 3 Section 1
        n_seen = (1 if self.vpp_orange_pose else 0) + (1 if self.vpp_green_pose else 0)
        n_in_zone = (1 if self.vpp_orange_in_zone else 0) + (1 if self.vpp_green_in_zone else 0)
        if hasattr(self, "lbl_vpp_orange_info"):
            if self.vpp_orange_pose:
                ox, oy, oz, _ = self.vpp_orange_pose
                badge = "🟢 TRONG Ô GẮP" if self.vpp_orange_in_zone else "🟡 Chờ ngoài ô gắp"
                self.lbl_vpp_orange_info.setText(f"   [{ox:.3f}, {oy:.3f}, {oz:.3f}] | {badge}")
                self.lbl_vpp_orange_info.setStyleSheet(
                    f"font-family: monospace; font-size: 10px; color: {'#34D399' if self.vpp_orange_in_zone else '#FBBF24'}; font-weight: bold;"
                )
            else:
                self.lbl_vpp_orange_info.setText("   ⚪ Không thấy trên băng tải")
                self.lbl_vpp_orange_info.setStyleSheet("font-family: monospace; font-size: 10px; color: #94A3B8;")

        if hasattr(self, "lbl_vpp_green_info"):
            if self.vpp_green_pose:
                gx, gy, gz, _ = self.vpp_green_pose
                badge = "🟢 TRONG Ô GẮP" if self.vpp_green_in_zone else "🟡 Chờ ngoài ô gắp"
                self.lbl_vpp_green_info.setText(f"   [{gx:.3f}, {gy:.3f}, {gz:.3f}] | {badge}")
                self.lbl_vpp_green_info.setStyleSheet(
                    f"font-family: monospace; font-size: 10px; color: {'#34D399' if self.vpp_green_in_zone else '#FBBF24'}; font-weight: bold;"
                )
            else:
                self.lbl_vpp_green_info.setText("   ⚪ Không thấy trên băng tải")
                self.lbl_vpp_green_info.setStyleSheet("font-family: monospace; font-size: 10px; color: #94A3B8;")

        if hasattr(self, "lbl_vpp_box_xyz") and hasattr(self, "lbl_vpp_box_state"):
            if box_state not in ("CARRIED", "MISSING") and self.vision_box_pose:
                bx, by, bz, _ = self.vision_box_pose
                name_tag = "Vuông Cam" if target_kind == "orange_cube" else "Trụ Xanh"
                self.lbl_vpp_box_xyz.setText(f"Ô gắp: {name_tag} [{bx:.3f}, {by:.3f}, {bz:.3f}] m")
            if box_state == "CARRIED":
                self.lbl_vpp_box_state.setText("🔵 Đang kẹp trên tay máy")
                self.lbl_vpp_box_state.setStyleSheet("font-size: 10px; font-weight: bold; color: #38BDF8;")
            elif n_seen > 0:
                self.lbl_vpp_box_state.setText(f"📡 Nhận diện {n_seen} phôi ({n_in_zone} trong Ô gắp)")
                self.lbl_vpp_box_state.setStyleSheet("font-size: 10px; font-weight: bold; color: #34D399;")
            else:
                self.lbl_vpp_box_state.setText("⚪ Chưa thấy phôi trên băng tải")
                self.lbl_vpp_box_state.setStyleSheet("font-size: 10px; color: #94A3B8;")

        if box_state == "CARRIED" and st.get("box_pose"):
            self.lbl_vis_box_status.setText("🔵 Đang được gắp (Theo tay kẹp)")
            self.lbl_vis_box_status.setStyleSheet("font-size: 10px; font-weight: bold; color: #38BDF8;")
        elif box_state == "OCCLUDED" and st.get("box_pose"):
            self.vision_box_pose = [float(v) for v in st["box_pose"][:4]]
            self.vision_box_ts = time.monotonic()
            dims = getattr(self, "vision_box_dims", None)
            if dims and len(dims) >= 3:
                yaw = self.vision_box_pose[3]
                self.lbl_vis_box_yaw.setText(
                    f"Yaw:{math.degrees(yaw):.0f}° | {dims[0]*100:.1f}×{dims[1]*100:.1f}×{dims[2]*100:.1f}cm"
                )
            self.lbl_vis_box_status.setText("🟡 Bị che — Giữ thông số đã lưu")
            self.lbl_vis_box_status.setStyleSheet("font-size: 10px; font-weight: bold; color: #FBBF24;")
        elif box_state == "MISSING":
            self.vision_box_pose = None
            self.vision_box_dims = None
            self.lbl_vis_box_xyz.setText("XYZ: ---, ---, --- m")
            self.lbl_vis_box_yaw.setText("Yaw: ---° | W×D×H: --- cm")
            self.lbl_vis_box_status.setText("⚪ Chờ phôi vào ranh giới Ô gắp")
            self.lbl_vis_box_status.setStyleSheet("font-size: 10px; color: #94A3B8;")
            if hasattr(self, "lbl_vpp_box_xyz"):
                self.lbl_vpp_box_xyz.setText("Ô gắp: Chờ phôi vào ranh giới [-0.465..-0.325]")

        if obs_state == "OCCLUDED":
            dims = getattr(self, "vision_obs_dims", None)
            if dims and len(dims) >= 3:
                self.lbl_vis_obs_dim.setText(
                    f"W×D×H: {dims[0]*100:.1f}×{dims[1]*100:.1f}×{dims[2]*100:.1f} cm"
                )
            obs_txt, obs_col = "Đồng bộ Cản (🔒 Lưu kích thước chuẩn)", "#FBBF24"
        elif obs_state == "OK":
            obs_txt, obs_col = "Đồng bộ Cản → NMPC (🔒 Đã lưu)", "#34D399"
        else:
            obs_txt, obs_col = "Đồng bộ Cản → NMPC", "#34D399"
        for chk in (getattr(self, "chk_vision_sync_obs", None), getattr(self, "chk_vision_sync_obs_tab6", None)):
            if chk is not None:
                chk.setText(obs_txt)
                chk.setStyleSheet(f"color: {obs_col}; font-weight: bold; font-size: 10px;")

    def on_toggle_vision_obs_sync(self, checked):
        self.vision_sync_obs = bool(checked)
        for chk in (getattr(self, "chk_vision_sync_obs", None), getattr(self, "chk_vision_sync_obs_tab6", None)):
            if chk is not None and chk.isChecked() != self.vision_sync_obs:
                chk.blockSignals(True)
                chk.setChecked(self.vision_sync_obs)
                chk.blockSignals(False)

    # ========================================================
    # TAB 6 VISION PICK & PLACE ENGINE (TEACH PLACE + UPWARD PARABOLIC ARC)
    # ========================================================
    def _quick_gripper_set(self, width_mm):
        self.grip_slider.setValue(int(width_mm))
        self.on_gripper_slider(int(width_mm))

    def _vpp_set_phase_ui(self, text, pct, color="#38BDF8"):
        key = (text, color)
        if getattr(self, "_last_vpp_phase_msg", None) != key:
            self._last_vpp_phase_msg = key
            self.lbl_vpp_phase.setText(text)
            self.lbl_vpp_phase.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 10px;")
        self.vpp_progress.setValue(int(max(0, min(100, pct))))

    def confirm_vpp_place_pose(self):
        px, py, pz = float(self.tcp_pos[0]), float(self.tcp_pos[1]), max(0.025, float(self.tcp_pos[2]))
        j6 = float(self.joint_angles[5]) if len(self.joint_angles) >= 6 else float(self.desired_j6_rad)
        self.place_pose_taught = [px, py, pz, j6]
        self.lbl_vpp_place_status.setText(
            f"🟢 Đã xác nhận điểm đặt: [{px:.3f}, {py:.3f}, {pz:.3f}] m | R = {math.degrees(j6):.1f}°"
        )
        self.lbl_vpp_place_status.setStyleSheet("color: #34D399; font-family: monospace; font-size: 10px; font-weight: bold;")

    def clear_vpp_place_pose(self):
        self.place_pose_taught = None
        self.lbl_vpp_place_status.setText(
            "⚪ Chưa xác nhận vị trí đặt — Hãy Jog tay máy đến điểm cần đặt rồi bấm Xác Nhận."
        )
        self.lbl_vpp_place_status.setStyleSheet("color: #FBBF24; font-family: monospace; font-size: 10px; font-weight: bold;")

    def _on_vpp_run_mode_changed(self, mode):
        self.vpp_run_mode = "auto" if mode == "auto" else "manual"
        if getattr(self, "conveyor_running", False):
            self.toggle_conveyor_feeder(False)
        if hasattr(self, "conv_ctrl_stack"):
            self.conv_ctrl_stack.setCurrentIndex(1 if self.vpp_run_mode == "manual" else 0)
        if hasattr(self, "chk_vpp_autofeed"):
            self.chk_vpp_autofeed.setEnabled(self.vpp_run_mode == "auto")
        if self.vpp_run_mode == "manual":
            self.vpp_auto_armed = False
            self.vpp_stable_pose = None
            if hasattr(self, "btn_vpp_start") and not self.vpp_active:
                self.btn_vpp_start.setText("▶ CHẠY MANUAL 1 LẦN (VÙNG TRỐNG)")
                self.btn_vpp_start.setEnabled(True)
            if not self.vpp_active and hasattr(self, "lbl_vpp_phase"):
                self._vpp_set_phase_ui(
                    "🖐️ Chế độ MANUAL 1 lần: Chạy Tiến/Lùi băng tải (4), Jog vào vùng trống & bấm Xác Nhận Điểm Đặt.",
                    0, "#38BDF8"
                )
        else:
            if hasattr(self, "btn_vpp_start") and not self.vpp_active:
                self.btn_vpp_start.setText("▶ CHẠY AUTO LIÊN TỤC (XẾP Ô 2×2)")
            if not self.vpp_active and hasattr(self, "lbl_vpp_phase"):
                self._vpp_set_phase_ui(
                    "🤖 Chế độ AUTO liên tục: Bấm Bắt Đầu để tự gắp phôi từ băng tải & phân loại vào Ô 2×2.",
                    0, "#34D399"
                )

    def _update_vpp_tray_counts_ui(self):
        if hasattr(self, "lbl_vpp_tray_counts"):
            self.lbl_vpp_tray_counts.setText(
                f"🧩 Ô 2×2 → Cam: {self.vpp_auto_orange_idx}/2 | Xanh: {self.vpp_auto_green_idx}/2"
            )

    def _next_auto_place_pose(self, kind):
        if kind == "orange_cube":
            idx = self.vpp_auto_orange_idx
        elif kind == "green_cyl":
            idx = self.vpp_auto_green_idx
        else:
            return None
        slots = self.vpp_auto_slots.get(kind, [])
        if idx >= len(slots):
            return None
        sx, sy, sz = slots[idx]
        # Orient wrist J6 at drop so cube's 4 edges align parallel to the square slot (0 deg world X/Y);
        # cylinder uses radial symmetry (j6 = 0.0) centered on circle marker.
        j6_place = self._compute_pick_j6(sx, sy, 0.0, kind=kind)
        return [float(sx), float(sy), float(sz), float(j6_place)]

    @staticmethod
    def _is_in_pick_boundary(bx, by):
        return (0.305 <= float(bx) <= 0.595) and (-0.465 <= float(by) <= -0.325)

    def _publish_conveyor_velocity(self, vel_mps: float):
        """Publish smooth physics belt speed via ROS 2 TrackController bridge + async gz_transport."""
        self.ros.publish_conveyor_vel(vel_mps)
        if getattr(self, "_gz_conv_pub", None) is not None and double_pb2 is not None:
            try:
                m = double_pb2.Double()
                m.data = float(vel_mps)
                self._gz_conv_pub.publish(m)
            except Exception:
                pass

    def _sync_conveyor_ui(self):
        running = bool(self.conveyor_running)
        fwd_active = running and self.conveyor_direction > 0
        rev_active = running and self.conveyor_direction < 0
        if hasattr(self, "btn_vpp_conveyor"):
            self.btn_vpp_conveyor.blockSignals(True)
            self.btn_vpp_conveyor.setChecked(running)
            self.btn_vpp_conveyor.setText(
                "⏸ Dừng Băng Tải (Đang chạy ▶)" if running
                else "🏭 Chạy Băng Tải (Đưa vào Ô gắp)"
            )
            self.btn_vpp_conveyor.blockSignals(False)
        if hasattr(self, "btn_vpp_conv_fwd"):
            self.btn_vpp_conv_fwd.blockSignals(True)
            self.btn_vpp_conv_fwd.setChecked(fwd_active)
            self.btn_vpp_conv_fwd.blockSignals(False)
        if hasattr(self, "btn_vpp_conv_rev"):
            self.btn_vpp_conv_rev.blockSignals(True)
            self.btn_vpp_conv_rev.setChecked(rev_active)
            self.btn_vpp_conv_rev.blockSignals(False)
        if hasattr(self, "lbl_vpp_conv_dir"):
            if fwd_active:
                self.lbl_vpp_conv_dir.setText(f"▶ Tiến (+{self.conveyor_speed_mps:.2f} m/s)")
                self.lbl_vpp_conv_dir.setStyleSheet("font-family: monospace; font-size: 10px; font-weight: bold; color: #34D399;")
            elif rev_active:
                self.lbl_vpp_conv_dir.setText(f"◀ Lùi (-{self.conveyor_speed_mps:.2f} m/s)")
                self.lbl_vpp_conv_dir.setStyleSheet("font-family: monospace; font-size: 10px; font-weight: bold; color: #FB923C;")
            else:
                self.lbl_vpp_conv_dir.setText("⏹ Băng tải: Đang dừng")
                self.lbl_vpp_conv_dir.setStyleSheet("font-family: monospace; font-size: 10px; font-weight: bold; color: #94A3B8;")

    def toggle_conveyor_direction(self, direction: int):
        """Manual conveyor forward (+1) / reverse (-1) button toggle."""
        d = 1 if int(direction) >= 0 else -1
        if self.conveyor_running and self.conveyor_direction == d:
            self.toggle_conveyor_feeder(False)
        else:
            self.toggle_conveyor_feeder(True, direction=d)

    def toggle_conveyor_feeder(self, checked=None, direction=None):
        if direction is not None:
            self.conveyor_direction = 1 if int(direction) >= 0 else -1
        elif checked is None and not self.conveyor_running:
            self.conveyor_direction = 1
        if checked is None:
            self.conveyor_running = not self.conveyor_running
        else:
            self.conveyor_running = bool(checked)
        self._conv_last_ts = time.monotonic()
        self._conv_last_pub_ts = 0.0
        if self.conveyor_running:
            self.vpp_stable_pose = None
            self.vpp_stable_ts = 0.0
            self._publish_conveyor_velocity(float(self.conveyor_direction) * float(self.conveyor_speed_mps))
        else:
            self._publish_conveyor_velocity(0.0)
        self._sync_conveyor_ui()

    def _step_conveyor_feeder(self):
        if not self.conveyor_running:
            return
        now = time.monotonic()
        if self._conv_last_ts <= 0.0:
            self._conv_last_ts = now
            return
        dt = now - self._conv_last_ts
        if dt < 0.04:
            return
        self._conv_last_ts = now

        signed_vel = float(self.conveyor_direction) * float(self.conveyor_speed_mps)
        items = getattr(self, "vpp_conveyor_items", None) or []
        ys = [float(it["pose"][1]) for it in items if it.get("pose")]
        if self.vpp_orange_pose is not None:
            ys.append(float(self.vpp_orange_pose[1]))
        if self.vpp_green_pose is not None:
            ys.append(float(self.vpp_green_pose[1]))

        if self.conveyor_direction > 0:
            # Front drop-off safety edge near wooden board Y=-0.30
            if any(-0.338 <= y <= -0.28 for y in ys):
                self.toggle_conveyor_feeder(False)
                return
            if self.vpp_run_mode == "auto":
                orange_ready = self.vpp_orange_pose is not None and -0.405 <= self.vpp_orange_pose[1] <= -0.325
                green_ready = self.vpp_green_pose is not None and -0.405 <= self.vpp_green_pose[1] <= -0.325
                if orange_ready or green_ready:
                    self.toggle_conveyor_feeder(False)
                    return
                waiting_any = any(-0.92 <= y < -0.345 for y in ys)
                if self.vision_box_ts > 0.0 and (now - self.vision_box_ts) < 1.5 and not waiting_any:
                    self.toggle_conveyor_feeder(False)
                    return
        else:
            # Back belt edge safety stop when reversing
            if any(y <= -0.90 for y in ys):
                self.toggle_conveyor_feeder(False)
                return

        # Re-publish belt velocity gently at 10 Hz for TrackController smoothness
        if (now - getattr(self, "_conv_last_pub_ts", 0.0)) >= 0.10:
            self._publish_conveyor_velocity(signed_vel)
            self._conv_last_pub_ts = now

    def _feed_next_conveyor_part(self):
        """Start smooth forward belt movement so the next waiting workpiece enters the Pick Boundary."""
        waiting_green = self.vpp_auto_green_idx < 2 and self.vpp_green_pose and self.vpp_green_pose[1] < -0.405
        waiting_orange = self.vpp_auto_orange_idx < 2 and self.vpp_orange_pose and self.vpp_orange_pose[1] < -0.405
        if waiting_green or waiting_orange:
            self.toggle_conveyor_feeder(True, direction=1)

    def _vpp_target_place_j6(self, pick_j6=0.0):
        base_pose = self.vpp_cycle_place_pose or self.place_pose_taught
        taught_j6 = base_pose[3] if base_pose else 0.0
        mode = self.cmb_vpp_place_rot.currentIndex() if hasattr(self, "cmb_vpp_place_rot") else 0
        if mode == 1:
            return 0.0
        if mode == 2:
            return float(pick_j6)
        if mode == 3:
            ang = taught_j6 + math.pi * 0.5
            while ang > math.pi:
                ang -= 2.0 * math.pi
            return ang
        return float(taught_j6)

    def preview_vpp_place_pose(self):
        if not self.place_pose_taught:
            QtWidgets.QMessageBox.warning(
                self, "Chưa có điểm đặt",
                "Bạn chưa xác nhận vị trí đặt vật.\nHãy Jog tay máy tới chỗ đặt rồi bấm '📍 XÁC NHẬN VỊ TRÍ ĐẶT'."
            )
            return
        self.is_running = False
        px, py, pz, _ = self.place_pose_taught
        j6_pl = self._vpp_target_place_j6(self.desired_j6_rad)
        hover_z = max(pz + 0.14, 0.22)
        cur_j6 = float(self.joint_angles[5]) if len(self.joint_angles) >= 6 else self.desired_j6_rad
        self.vpp_active = True
        self.btn_vpp_start.setEnabled(False)
        self.btn_vpp_stop.setEnabled(True)
        self._start_upward_arc(
            [self.tcp_pos[0], self.tcp_pos[1], self.tcp_pos[2], cur_j6],
            [px, py, hover_z, j6_pl],
            next_state="PREVIEW_DESCEND",
            peak_z_min=self.sp_vpp_arc_z.value(),
        )
        self._vpp_set_phase_ui("🎯 Đang bay vòng cung lên thử tới vị trí đặt đã xác nhận...", 50, "#38BDF8")

    def _start_upward_arc(self, start_xyzj, end_xyzj, next_state, peak_z_min=0.36):
        self.vpp_state = "ARC_MOVE"
        self.vpp_state_ts = time.monotonic()
        self.vpp_arc_start = list(start_xyzj)
        self.vpp_arc_end = list(end_xyzj)
        self.vpp_arc_s = 0.0
        self.vpp_arc_next_state = next_state

        obs_top = self.vision_obs_info[2] if (self.vision_obs_info and len(self.vision_obs_info) >= 3) else 0.25
        z0, z1 = start_xyzj[2], end_xyzj[2]
        self.vpp_arc_peak_z = max(float(peak_z_min), float(obs_top) + 0.14, max(z0, z1) + 0.10)
        dist_xy = math.hypot(end_xyzj[0] - start_xyzj[0], end_xyzj[1] - start_xyzj[1])
        speed_factor = max(0.25, self.speed_slider.value() / 100.0)
        self.vpp_arc_dur = max(1.6, dist_xy / (0.20 * speed_factor))

    def _launch_vpp_cycle(self, box, kind, place_pose):
        bx, by, bz, yaw_box = [float(v) for v in box[:4]]
        dims = getattr(self, "vision_box_dims", None)
        box_h = float(dims[2]) if (dims and len(dims) >= 3) else 0.06

        # Auto-enable obstacle sync if Vision obstacle is present (rail mode pushes sync; dynamic uses 30Hz dynamic node)
        if self.vision_obs_info is not None:
            self.on_toggle_vision_obs_sync(True)
            if self.sim_scenario != "dynamic":
                ox, oy, oz_top, orad = self.vision_obs_info
                safe_r = max(0.05, min(0.20, orad + 0.02))
                self.radar.update_obstacle(ox, oy, safe_r, self.warn_margin, self.danger_margin)
                self.ros.publish_obstacle(ox, oy, oz_top * 0.6, z_top=oz_top, radius=orad)

        j6_pick = 0.0 if kind == "green_cyl" else self._compute_pick_j6(bx, by, yaw_box, kind=kind, dims=dims)
        self.desired_j6_rad = float(j6_pick)
        dz_offset = float(self.sp_vpp_pick_dz.value()) if hasattr(self, "sp_vpp_pick_dz") else -0.018
        pick_z = max(0.020, min(0.18, float(bz) + dz_offset))
        self.vpp_latched_box = [bx, by, pick_z, yaw_box, float(j6_pick), box_h]
        self.vpp_cycle_kind = kind
        self.vpp_cycle_place_pose = [float(place_pose[0]), float(place_pose[1]), float(place_pose[2]), float(place_pose[3])]

        if self.vpp_grasp_locked and self.vpp_confirmed_grasp_width is not None:
            grasp_w = int(self.vpp_confirmed_grasp_width)
            grasp_f = float(self.vpp_confirmed_grasp_force if self.vpp_confirmed_grasp_force is not None else self.gripper_force)
        else:
            grasp_w = int(self.sp_vpp_grasp_width.value()) if hasattr(self, "sp_vpp_grasp_width") else 55
            grasp_f = float(self.gripper_force)
        hold_delay_s = float(self.sp_vpp_grasp_delay.value()) if hasattr(self, "sp_vpp_grasp_delay") else 1.2
        self.vpp_latched_grasp = (grasp_w, grasp_f, hold_delay_s)

        self.is_running = False
        self.vpp_active = True
        self.vpp_state = "PRE_OPEN"
        self.vpp_state_ts = time.monotonic()
        self.vpp_settle_ts = 0.0
        self.btn_vpp_start.setEnabled(False)
        self.btn_vpp_stop.setEnabled(True)

        self._quick_gripper_set(85)
        tag = "Vuông Cam" if kind == "orange_cube" else ("Trụ Xanh" if kind == "green_cyl" else "Manual")
        self._vpp_set_phase_ui(
            f"1/8 Mở kẹp [{tag} | J6={math.degrees(j6_pick):+.1f}°] [{bx:.3f}, {by:.3f}, {pick_z:.3f}] → [{place_pose[0]:.3f}, {place_pose[1]:.3f}]",
            8, "#38BDF8"
        )

    def _step_vpp_auto_monitor(self):
        if not self.vpp_auto_armed or self.vpp_active:
            return
        if self.conveyor_running:
            self.vpp_stable_pose = None
            self.vpp_stable_ts = 0.0
            self._vpp_set_phase_ui(
                "⏳ Băng tải đang chạy — chờ phôi vào Ô gắp & băng tải dừng hẳn...",
                5, "#FBBF24"
            )
            return
        if not self.vision_box_pose or (time.monotonic() - self.vision_box_ts) > 1.5:
            self.vpp_stable_pose = None
            self.vpp_stable_ts = 0.0
            return
        if getattr(self, "vision_box_state", "OK") != "OK":
            self.vpp_stable_pose = None
            self.vpp_stable_ts = 0.0
            return
        bx, by, bz, yaw = [float(v) for v in self.vision_box_pose[:4]]
        if not self._is_in_pick_boundary(bx, by):
            self.vpp_stable_pose = None
            self.vpp_stable_ts = 0.0
            self._vpp_set_phase_ui(
                "⏳ Phôi ngoài ranh giới vạch Ô gắp — chưa được phép gắp...",
                5, "#94A3B8"
            )
            return

        kind = self.vpp_active_target_kind or "orange_cube"
        place_pose = self._next_auto_place_pose(kind)
        if place_pose is None:
            return

        now = time.monotonic()
        yaw_diff = 0.0
        if self.vpp_stable_pose is not None and kind != "green_cyl":
            yaw_diff = abs(((yaw - self.vpp_stable_pose[3] + math.pi * 0.25) % (math.pi * 0.5)) - math.pi * 0.25)
        if (
            self.vpp_stable_pose is None
            or math.hypot(bx - self.vpp_stable_pose[0], by - self.vpp_stable_pose[1]) > 0.008
            or yaw_diff > math.radians(6.0)
        ):
            self.vpp_stable_pose = [bx, by, bz, yaw]
            self.vpp_stable_ts = now
            self._vpp_set_phase_ui(
                f"⏳ Phôi ổn định trong Ô gắp: 0.0 / {self.vpp_dwell_sec:.1f} s...",
                8, "#38BDF8"
            )
            return

        held = now - self.vpp_stable_ts
        if held < float(self.vpp_dwell_sec):
            pct = int(8 + 14 * min(1.0, held / float(self.vpp_dwell_sec)))
            self._vpp_set_phase_ui(
                f"⏳ Xác nhận phôi đứng yên trong Ô gắp: {held:.1f} / {self.vpp_dwell_sec:.1f} s...",
                pct, "#38BDF8"
            )
            return

        self.vpp_stable_pose = None
        self.vpp_stable_ts = 0.0
        self._launch_vpp_cycle([bx, by, bz, yaw], kind, place_pose)

    def _finish_vpp_cycle(self, went_home=True):
        if went_home:
            self.send_joint_pose([0.0, 0.0, 1.5708, 0.0, 1.5708, 0.0])
        self.vpp_active = False
        self.vpp_state = "IDLE"
        self.vpp_stable_pose = None

        if self.vpp_auto_armed:
            if self.vpp_cycle_kind == "orange_cube":
                self.vpp_auto_orange_idx = min(2, self.vpp_auto_orange_idx + 1)
            elif self.vpp_cycle_kind == "green_cyl":
                self.vpp_auto_green_idx = min(2, self.vpp_auto_green_idx + 1)
            self._update_vpp_tray_counts_ui()

            if self.vpp_auto_orange_idx >= 2 and self.vpp_auto_green_idx >= 2:
                self.vpp_auto_armed = False
                self.btn_vpp_start.setEnabled(True)
                self.btn_vpp_stop.setEnabled(False)
                self._vpp_set_phase_ui("🎉 Hoàn tất! Đã xếp đầy toàn bộ Ô 2×2 (2 Cam + 2 Xanh).", 100, "#34D399")
            else:
                if self.vpp_conveyor_autofeed:
                    self._feed_next_conveyor_part()
                self.btn_vpp_start.setEnabled(False)
                self.btn_vpp_stop.setEnabled(True)
                self._vpp_set_phase_ui(
                    f"✓ Đã xếp vào Ô 2×2 — Đang chờ phôi tiếp theo trên băng tải...",
                    12, "#34D399"
                )
        else:
            self.btn_vpp_start.setEnabled(True)
            self.btn_vpp_stop.setEnabled(False)
            self._vpp_set_phase_ui("✓ Hoàn thành 1 chu trình Manual Gắp – Né – Thả!", 100, "#34D399")

    def start_vision_pick_place(self):
        if self.vpp_run_mode == "auto":
            if self.vpp_auto_orange_idx >= 2 and self.vpp_auto_green_idx >= 2:
                QtWidgets.QMessageBox.information(
                    self, "Ô 2×2 đã đầy",
                    "Cả 4 vị trí trong Ô 2×2 (2 Cam + 2 Xanh) đều đã đầy.\nHãy bấm '🔄 Reset Phôi' để làm mới khay."
                )
                return
            self.vpp_auto_armed = True
            self.vpp_stable_pose = None
            self.vpp_stable_ts = 0.0
            self.btn_vpp_start.setEnabled(False)
            self.btn_vpp_stop.setEnabled(True)
            in_zone_now = (
                self.vision_box_pose
                and (time.monotonic() - self.vision_box_ts) <= 1.5
                and getattr(self, "vision_box_state", "MISSING") == "OK"
                and self._is_in_pick_boundary(self.vision_box_pose[0], self.vision_box_pose[1])
            )
            if not self.conveyor_running and not in_zone_now and self.vpp_conveyor_autofeed:
                self._feed_next_conveyor_part()
            self._vpp_set_phase_ui(
                f"🤖 AUTO SẴN SÀNG: Chờ băng tải dừng & giữ phôi trong Ô gắp {self.vpp_dwell_sec:.1f}s...",
                5, "#34D399"
            )
            self._step_vpp_auto_monitor()
            return

        # Manual 1-Shot mode: refuse while conveyor is running, otherwise pick selected in-zone workpiece
        if self.conveyor_running:
            QtWidgets.QMessageBox.warning(
                self, "Băng tải đang chạy",
                "Không thể gắp khi băng tải vẫn đang hoạt động!\n"
                "Hãy bấm '⏸ Dừng Băng Tải' trước khi gắp."
            )
            return
        self.vpp_auto_armed = False
        chosen_kind = self.vpp_active_target_kind or "manual"
        box = None
        if self.vpp_manual_target_pref == "orange_cube" and self.vpp_orange_pose and self._is_in_pick_boundary(*self.vpp_orange_pose[:2]):
            box = self.vpp_orange_pose
            chosen_kind = "orange_cube"
        elif self.vpp_manual_target_pref == "green_cyl" and self.vpp_green_pose and self._is_in_pick_boundary(*self.vpp_green_pose[:2]):
            box = self.vpp_green_pose
            chosen_kind = "green_cyl"
        elif self.vision_box_pose and (time.monotonic() - self.vision_box_ts) <= 3.0 and self._is_in_pick_boundary(*self.vision_box_pose[:2]):
            box = self.vision_box_pose
        else:
            QtWidgets.QMessageBox.warning(
                self, "Chưa có phôi trong Ô gắp",
                "Không có phôi nào đang nằm bên trong ranh giới Ô gắp trên băng tải (Y = -0.495..-0.295).\n"
                "Hãy bấm '🏭 Chạy Băng Tải' để đưa phôi vào đúng ranh giới Ô gắp trước!"
            )
            return
        if not self.place_pose_taught:
            QtWidgets.QMessageBox.warning(
                self, "Thiếu vị trí đặt vật",
                "Bạn chưa xác nhận vị trí đặt vật!\n"
                "Hãy dùng cụm Jog ở bước (2) đưa tay máy vào Vùng Trống rồi bấm '📍 XÁC NHẬN VỊ TRÍ ĐẶT'."
            )
            return

        dims = getattr(self, "vision_box_dims", None)
        box_h = float(dims[2]) if (dims and len(dims) >= 3) else 0.06
        if box_h < 0.045:
            ans = QtWidgets.QMessageBox.question(
                self, "Cảnh báo phôi có thể đang bị đổ",
                f"Camera đo chiều cao phôi hiện tại chỉ là {box_h*100:.1f} cm (chuẩn ~6.0 cm).\n"
                "Bạn vẫn muốn tiếp tục gắp theo tọa độ Camera hiện tại chứ?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            if ans != QtWidgets.QMessageBox.Yes:
                return

        self._launch_vpp_cycle(box, chosen_kind, self.place_pose_taught)

    def stop_vision_pick_place(self):
        self.vpp_active = False
        self.vpp_auto_armed = False
        self.vpp_state = "IDLE"
        self.vpp_stable_pose = None
        self.btn_vpp_start.setEnabled(True)
        self.btn_vpp_stop.setEnabled(False)
        self._vpp_set_phase_ui("⏹ Đã dừng chu trình / tắt chế độ Auto Vision.", 0, "#FBBF24")

    def step_vision_pick_place(self):
        if not self.vpp_active:
            return
        now = time.monotonic()
        elapsed = now - self.vpp_state_ts

        if self.vpp_state == "PRE_OPEN":
            obs_info = self.vision_obs_info
            obs_top = float(obs_info[2]) if (obs_info and len(obs_info) >= 3) else 0.26
            near_obs = (
                obs_info is not None
                and math.hypot(self.tcp_pos[0] - float(obs_info[0]), self.tcp_pos[1] - float(obs_info[1])) < 0.18
                and self.tcp_pos[2] < obs_top + 0.075
            )
            if near_obs:
                pre_z = max(float(self.tcp_pos[2]), obs_top + 0.085)
                self.ros.publish_target(0.368, 0.006, pre_z, yaw=self.desired_j6_rad)
            if elapsed >= (0.95 if near_obs else 0.55) and (not near_obs or self.tcp_pos[2] >= obs_top + 0.055 or elapsed >= 1.5):
                bx, by, pick_z, _, j6_pick, _ = self.vpp_latched_box
                pick_hover_z = max(pick_z + 0.16, 0.24)
                cur_j6 = float(self.joint_angles[5]) if len(self.joint_angles) >= 6 else self.desired_j6_rad
                self._start_upward_arc(
                    [self.tcp_pos[0], self.tcp_pos[1], self.tcp_pos[2], cur_j6],
                    [bx, by, pick_hover_z, j6_pick],
                    next_state="DESCEND_PICK",
                    peak_z_min=self.sp_vpp_arc_z.value(),
                )
                self._vpp_set_phase_ui("2/8 Bay vòng cung lên tiếp cận phía trên vật...", 20, "#38BDF8")

        elif self.vpp_state == "ARC_MOVE":
            x0, y0, z0, j0 = self.vpp_arc_start
            x1, y1, z1, j1 = self.vpp_arc_end
            # Evaluate current carrot position on upward parabola
            s = min(1.0, self.vpp_arc_s)
            h_extra = max(0.05, self.vpp_arc_peak_z - 0.5 * (z0 + z1))
            cx = (1.0 - s) * x0 + s * x1
            cy = (1.0 - s) * y0 + s * y1
            cz = max(0.025, (1.0 - s) * z0 + s * z1 + 4.0 * s * (1.0 - s) * h_extra)
            cj6 = (1.0 - s) * j0 + s * j1

            # Pause carrot progression if robot lags > 6.5 cm so trajectory sticks to upward arc
            lag = math.hypot(math.hypot(self.tcp_pos[0] - cx, self.tcp_pos[1] - cy), self.tcp_pos[2] - cz)
            if lag < 0.065 or elapsed > self.vpp_arc_dur * 2.2:
                self.vpp_arc_s = min(1.0, self.vpp_arc_s + 0.02 / self.vpp_arc_dur)

            self.desired_j6_rad = cj6
            self.ros.publish_target(cx, cy, cz, yaw=cj6)
            self.radar.update_tcp(cx, cy, cz)

            end_dist = math.hypot(math.hypot(self.tcp_pos[0] - x1, self.tcp_pos[1] - y1), self.tcp_pos[2] - z1)
            if (self.vpp_arc_s >= 1.0 and end_dist < 0.040) or elapsed > (self.vpp_arc_dur * 2.5 + 3.0):
                nxt = self.vpp_arc_next_state
                self.vpp_state = nxt
                self.vpp_state_ts = now
                self.vpp_settle_ts = 0.0
                if nxt == "PREVIEW_DONE":
                    self.vpp_active = False
                    self.btn_vpp_start.setEnabled(True)
                    self.btn_vpp_stop.setEnabled(False)
                    self._vpp_set_phase_ui("✓ Đã tới điểm đặt thử thành công!", 100, "#34D399")
                elif nxt == "DONE":
                    self._finish_vpp_cycle(went_home=True)

        elif self.vpp_state == "PREVIEW_DESCEND":
            px, py, pz, _ = self.place_pose_taught
            j6_pl = self._vpp_target_place_j6(self.desired_j6_rad)
            self.desired_j6_rad = j6_pl
            self.ros.publish_target(px, py, pz, yaw=j6_pl)
            dist = math.hypot(math.hypot(self.tcp_pos[0] - px, self.tcp_pos[1] - py), self.tcp_pos[2] - pz)
            if dist < 0.030 or elapsed > 3.5:
                self.vpp_active = False
                self.vpp_state = "IDLE"
                self.btn_vpp_start.setEnabled(True)
                self.btn_vpp_stop.setEnabled(False)
                self._vpp_set_phase_ui("✓ Đã tới đúng vị trí đặt thử!", 100, "#34D399")

        elif self.vpp_state == "DESCEND_PICK":
            bx, by, pick_z, _, j6_pick, _ = self.vpp_latched_box
            self.desired_j6_rad = j6_pick
            self.ros.publish_target(bx, by, pick_z, yaw=j6_pick)
            dist_xy = math.hypot(self.tcp_pos[0] - bx, self.tcp_pos[1] - by)
            dz_err = abs(self.tcp_pos[2] - pick_z)
            cur_j6 = float(self.joint_angles[5]) if len(self.joint_angles) >= 6 else j6_pick
            j6_err = abs(((cur_j6 - j6_pick + math.pi) % (2.0 * math.pi)) - math.pi)
            self._vpp_set_phase_ui(f"3/8 Hạ thẳng xuống 1/2 chiều cao vật (Z={pick_z:.3f}m)...", 34, "#38BDF8")
            if (dist_xy < 0.020 and dz_err < 0.012 and j6_err < 0.08) or elapsed > 4.0:
                self.vpp_settle_ts += 0.02
                if self.vpp_settle_ts >= 0.30 or elapsed > 4.0:
                    self.vpp_state = "GRASP_CLOSE"
                    self.vpp_state_ts = now
                    target_w, force_n, _ = self.vpp_latched_grasp
                    self._vpp_set_phase_ui(
                        f"4/8 Đóng kẹp Robotiq 2F-85 tới {target_w} mm (Lực: {force_n:.0f} N)...",
                        44, "#FBBF24"
                    )

        elif self.vpp_state == "GRASP_CLOSE":
            bx, by, pick_z, _, j6_pick, _ = self.vpp_latched_box
            self.ros.publish_target(bx, by, pick_z, yaw=j6_pick)
            target_w, force_n, _ = self.vpp_latched_grasp
            force_n = max(10.0, float(force_n))
            # Physical finger closing time (URDF vel limit 0.5 rad/s takes ~1.0s); gentle force closes slower
            ramp_dur = max(1.0, min(1.8, 50.0 / force_n * 1.15))
            ratio = min(1.0, elapsed / ramp_dur)
            w_cmd = int(round(85.0 - ratio * (85.0 - target_w)))
            self._quick_gripper_set(w_cmd)
            if elapsed >= ramp_dur:
                self._quick_gripper_set(target_w)
                self.vpp_state = "GRASP_HOLD"
                self.vpp_state_ts = now
                self._vpp_set_phase_ui(
                    f"4/8 Đã đóng tới {target_w} mm — Chờ giữ ổn định kẹp...",
                    52, "#FBBF24"
                )

        elif self.vpp_state == "GRASP_HOLD":
            bx, by, pick_z, _, j6_pick, _ = self.vpp_latched_box
            self.ros.publish_target(bx, by, pick_z, yaw=j6_pick)
            target_w, force_n, hold_delay_s = self.vpp_latched_grasp
            self._quick_gripper_set(target_w)
            rem = max(0.0, hold_delay_s - elapsed)
            self._vpp_set_phase_ui(
                f"4/8 Giữ chắc {target_w} mm (Lực {force_n:.0f} N) — chờ ổn định {rem:.1f}s...",
                54, "#FBBF24"
            )
            if elapsed >= hold_delay_s:
                self.vpp_state = "LIFT_VERTICAL"
                self.vpp_state_ts = now
                self._vpp_set_phase_ui("5/8 Nhấc thẳng đứng khỏi mặt bàn...", 58, "#38BDF8")

        elif self.vpp_state == "LIFT_VERTICAL":
            bx, by, pick_z, _, j6_pick, _ = self.vpp_latched_box
            lift_z = max(pick_z + 0.16, 0.25)
            self.ros.publish_target(bx, by, lift_z, yaw=j6_pick)
            dist = math.hypot(math.hypot(self.tcp_pos[0] - bx, self.tcp_pos[1] - by), self.tcp_pos[2] - lift_z)
            if dist < 0.035 or elapsed > 3.0:
                px, py, pz, _ = (self.vpp_cycle_place_pose or self.place_pose_taught)
                place_hover_z = max(pz + 0.15, 0.23)
                j6_pl = self._vpp_target_place_j6(j6_pick)
                self._start_upward_arc(
                    [bx, by, lift_z, j6_pick],
                    [px, py, place_hover_z, j6_pl],
                    next_state="DESCEND_PLACE",
                    peak_z_min=self.sp_vpp_arc_z.value(),
                )
                self._vpp_set_phase_ui("6/8 Bay vòng cung đi lên vượt cản sang điểm đặt...", 72, "#10B981")

        elif self.vpp_state == "DESCEND_PLACE":
            px, py, pz, _ = (self.vpp_cycle_place_pose or self.place_pose_taught)
            j6_pl = self._vpp_target_place_j6(self.vpp_latched_box[4])
            target_pz = max(0.026, pz)
            self.desired_j6_rad = j6_pl
            self.ros.publish_target(px, py, target_pz, yaw=j6_pl)
            dist = math.hypot(math.hypot(self.tcp_pos[0] - px, self.tcp_pos[1] - py), self.tcp_pos[2] - target_pz)
            cur_j6 = float(self.joint_angles[5]) if len(self.joint_angles) >= 6 else j6_pl
            j6_err = abs(((cur_j6 - j6_pl + math.pi) % (2.0 * math.pi)) - math.pi)
            self._vpp_set_phase_ui(
                f"7/8 Hạ & xoay khớp J6={math.degrees(j6_pl):+.1f}° đặt đúng vào ô...",
                84, "#38BDF8"
            )
            if (dist < 0.022 and j6_err < 0.045) or elapsed > 4.0:
                self.vpp_settle_ts += 0.02
                if self.vpp_settle_ts >= 0.25 or elapsed > 4.0:
                    self.vpp_state = "RELEASE_OPEN"
                    self.vpp_state_ts = now
                    self._quick_gripper_set(85)
                    self._vpp_set_phase_ui("7/8 Mở kẹp thả vật...", 90, "#FBBF24")

        elif self.vpp_state == "RELEASE_OPEN":
            px, py, pz, _ = (self.vpp_cycle_place_pose or self.place_pose_taught)
            j6_pl = self._vpp_target_place_j6(self.vpp_latched_box[4])
            self.ros.publish_target(px, py, max(0.030, pz), yaw=j6_pl)
            if elapsed >= 0.85:
                self.vpp_state = "RETRACT_PLACE"
                self.vpp_state_ts = now
                self._vpp_set_phase_ui("8/8 Nhấc lên an toàn sau khi thả...", 95, "#38BDF8")

        elif self.vpp_state == "RETRACT_PLACE":
            px, py, pz, _ = (self.vpp_cycle_place_pose or self.place_pose_taught)
            j6_pl = self._vpp_target_place_j6(self.vpp_latched_box[4])
            place_hover_z = max(pz + 0.15, 0.24)
            self.ros.publish_target(px, py, place_hover_z, yaw=j6_pl)
            dist = math.hypot(math.hypot(self.tcp_pos[0] - px, self.tcp_pos[1] - py), self.tcp_pos[2] - place_hover_z)
            if dist < 0.038 or elapsed > 2.5:
                if self.chk_vpp_return_home.isChecked():
                    self._start_upward_arc(
                        [px, py, place_hover_z, j6_pl],
                        [0.368, 0.006, 0.236, 0.0],
                        next_state="DONE",
                        peak_z_min=self.sp_vpp_arc_z.value(),
                    )
                    self._vpp_set_phase_ui("8/8 Bay vòng cung về HOME an toàn...", 98, "#10B981")
                else:
                    self._finish_vpp_cycle(went_home=False)

    def _require_vision_box(self):
        if not self.vision_box_pose or (time.monotonic() - self.vision_box_ts) > 3.0:
            QtWidgets.QMessageBox.warning(
                self, "Chưa có dữ liệu Vision",
                "Chưa nhận diện được Hộp Vàng từ Camera D435 trong 3 giây gần nhất.\n"
                "Hãy đảm bảo Gazebo đang chạy và bạn đã bấm '▶ Bật Vision'."
            )
            return None
        return self.vision_box_pose

    def record_vision_box_to_tree(self):
        box = self._require_vision_box()
        if box is None:
            return
        bx, by, bz, yaw = box
        grasp_z = max(0.022, bz - 0.018)
        hover_id = len(self.program_steps) + 1
        self.program_steps.append({
            "id": hover_id,
            "type": "MoveNMPC",
            "target": "Vision_Box_Hover",
            "params": f"[{bx:.3f}, {by:.3f}, 0.220, {yaw:.3f}]",
            "desc": "Tiếp cận phía trên hộp từ Camera D435"
        })
        self.program_steps.append({
            "id": hover_id + 1,
            "type": "MoveNMPC",
            "target": "Vision_Box_Grasp",
            "params": f"[{bx:.3f}, {by:.3f}, {grasp_z:.3f}, {yaw:.3f}]",
            "desc": f"Hạ xuống tâm kẹp theo Vision (Yaw={math.degrees(yaw):.1f}°)"
        })
        self.refresh_tree_cards()
        self.update_tracker_display()
        self.tabs.setCurrentWidget(self.tab_builder)

    def start_vision_process(self):
        vis_proc = self.processes.get('vision')
        if vis_proc and vis_proc.poll() is None:
            return
        try:
            subprocess.run("pkill -9 -f 'dsr_vision_node'", shell=True, timeout=2)
            time.sleep(0.2)
        except Exception:
            pass
        cmd = self._build_ros_shell_cmd("ros2 run dsr_m0609_nmpc dsr_vision_node.py")
        p = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
        self.processes['vision'] = p

    def stop_vision_process(self):
        p = self.processes.pop('vision', None)
        if p:
            try:
                os.killpg(os.getpgid(p.pid), 9)
            except Exception:
                pass
        try:
            subprocess.run("pkill -9 -f 'dsr_vision_node'", shell=True, timeout=2)
        except Exception:
            pass
        self.lbl_vision_status.setText("🔴 ĐÃ DỪNG")
        self.lbl_vision_status.setStyleSheet("color: #EF4444; font-weight: bold; font-family: monospace;")
        self.btn_start_vision.setEnabled(True)
        self.btn_stop_vision.setEnabled(False)

    def confirm_and_exit_system(self):
        """Cleanly exit GUI and terminate ALL Gazebo & ROS 2 background nodes"""
        reply = QtWidgets.QMessageBox.question(
            self, "Xác nhận thoát hệ thống",
            "Bạn có chắc muốn đóng Teach Pendant và DỪNG TOÀN BỘ hệ thống (Gazebo, NMPC Node, Vision Node, ROS 2 Bridge)?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.terminate_all_background_nodes()
            QtWidgets.QApplication.quit()

    def terminate_all_background_nodes(self):
        """Kills any spawned process groups and executes pkill cleanup"""
        # Kill tracked Popen process groups
        for p in self.processes.values():
            try:
                os.killpg(os.getpgid(p.pid), 9)
            except Exception:
                pass
        self.processes.clear()

        # Full system sweep pkill to guarantee no ghost processes remain running
        try:
            subprocess.run(
                "pkill -9 -f 'gazebo|gz sim|gz-sim|m0609_nmpc_node|dsr_vision_node|dynamic_obstacle_node|robot_state_publisher|ros_gz_bridge|dsr_m0609_nmpc'",
                shell=True, timeout=3
            )
        except Exception:
            pass

    def closeEvent(self, event):
        reply = QtWidgets.QMessageBox.question(
            self, "Xác nhận thoát hệ thống",
            "Đóng Teach Pendant và DỪNG TOÀN BỘ hệ thống (Gazebo, NMPC, ROS 2)?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.terminate_all_background_nodes()
            event.accept()
        else:
            event.ignore()

    def apply_dark_theme(self):
        theme_qss = """
            QMainWindow { background-color: #080C14; }
            QWidget { color: #E2E8F0; font-family: 'Segoe UI', Arial, sans-serif; }
            
            QDialog, QMessageBox, QInputDialog {
                background-color: #0F172A;
                color: #F8FAFC;
                border: 1px solid #334155;
            }
            QMessageBox QLabel, QInputDialog QLabel, QDialog QLabel {
                color: #F8FAFC;
                background-color: transparent;
                font-size: 12px;
            }
            QMessageBox QPushButton, QInputDialog QPushButton {
                background-color: #0284C7;
                color: #FFFFFF;
                border: 1px solid #0369A1;
                border-radius: 6px;
                padding: 6px 18px;
                min-width: 65px;
                font-weight: bold;
            }
            QMessageBox QPushButton:hover, QInputDialog QPushButton:hover {
                background-color: #0369A1;
            }

            QFrame#headerFrame {
                background-color: #0B1220;
                border: 1px solid #1E293B;
                border-radius: 8px;
            }
            QFrame#cardFrame {
                background-color: #101726;
                border: 1px solid #1E293B;
                border-radius: 8px;
                padding: 6px;
            }
            
            QTabWidget::pane {
                border: 1px solid #1E293B;
                background-color: #0A101C;
                border-radius: 8px;
            }
            QTabBar::tab {
                background-color: #0E1524;
                color: #94A3B8;
                font-weight: bold;
                font-size: 11px;
                padding: 5px 8px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #131D31;
                color: #38BDF8;
                border-bottom: 2px solid #38BDF8;
            }
            
            QGroupBox {
                border: 1px solid #1E293B;
                border-radius: 8px;
                margin-top: 10px;
                font-weight: bold;
                font-size: 11px;
                color: #94A3B8;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            
            QPushButton {
                background-color: #1E293B;
                color: #E2E8F0;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #334155; }
            QPushButton:pressed { background-color: #0284C7; }
            
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background-color: #101726;
                color: #E2E8F0;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 4px;
                font-size: 11px;
            }
            
            QSlider::groove:horizontal {
                height: 6px;
                background: #1E293B;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #0284C7;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #38BDF8;
                width: 14px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 7px;
            }
        """
        self.setStyleSheet(theme_qss)
        app = QtWidgets.QApplication.instance()
        if app:
            app.setStyleSheet(theme_qss)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = TeachPendantApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
