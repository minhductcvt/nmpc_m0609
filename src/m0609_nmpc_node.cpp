#include <memory>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

// acados solver header
#include "acados_solver_ocp_m0609_kinematic_nmpc_41c0c9b9.h"
#include "m0609_kinematic_nmpc_cost.h"

using namespace std::chrono_literals;

class M0609NmpcNode : public rclcpp::Node
{
public:
  M0609NmpcNode()
  : Node("m0609_nmpc_node"),
    current_q_{0.0, 0.0, 1.5708, 0.0, 1.5708, 0.0},
    current_dq_{0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    ref_ee_pos_{0.368, 0.006, 0.236}, // initial target matches orthogonal HOME_SAFE pose with SVM (+25.6mm)
    target_joint_pos_{0.0, 0.0, 1.5708, 0.0, 1.5708, 0.0},
    desired_j6_{0.0},
    obs_pos_{0.50, 0.0, 0.13},       // matches Gazebo yellow obstacle pillar on rail
    obs_vel_{0.0, 0.0},
    obs_radius_{0.05},              // 5 cm actual pillar radius (10 cm diameter)
    obs_z_top_{0.26},               // 26 cm actual pillar top height (25-30 cm)
    has_joint_state_(false),
    has_target_joint_(false),
    need_warmstart_reset_(true),
    last_obs_vel_time_(this->now())
  {
    RCLCPP_INFO(this->get_logger(), "Initializing Doosan M0609 C++ NMPC Node (Capsule + Gripper Envelope + Anti-Deadlock)...");

    // 1. Initialize acados solver capsule
    capsule_ = ocp_m0609_kinematic_nmpc_41c0c9b9_acados_create_capsule();
    int status = ocp_m0609_kinematic_nmpc_41c0c9b9_acados_create(capsule_);
    if (status != 0) {
      RCLCPP_ERROR(this->get_logger(), "Failed to create acados OCP solver, status: %d", status);
    } else {
      RCLCPP_INFO(this->get_logger(), "acados RTI NMPC Solver created successfully!");
    }

    // 2. Subscribers
    joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/dsr01/joint_states", 10,
      std::bind(&M0609NmpcNode::jointStateCallback, this, std::placeholders::_1)
    );
    joint_state_sub_fallback_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", 10,
      std::bind(&M0609NmpcNode::jointStateCallback, this, std::placeholders::_1)
    );

    // Sub End-Effector target pose
    target_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/nmpc/target_ee_pose", 10,
      [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        if (has_target_joint_) {
          need_warmstart_reset_ = true;
        }
        has_target_joint_ = false; // switch to Cartesian NMPC mode
        ref_ee_pos_[0] = msg->pose.position.x;
        ref_ee_pos_[1] = msg->pose.position.y;
        ref_ee_pos_[2] = msg->pose.position.z;
        // If orientation is provided (e.g., yaw for gripper orientation)
        double qz = msg->pose.orientation.z;
        double qw = msg->pose.orientation.w;
        if (std::abs(qz) > 1e-3 || std::abs(qw - 1.0) > 1e-3) {
          desired_j6_ = 2.0 * std::atan2(qz, qw);
        }
        RCLCPP_INFO_THROTTLE(
          this->get_logger(), *this->get_clock(), 1000,
          "New EE Target: [%.3f, %.3f, %.3f] | J6: %.2f",
          ref_ee_pos_[0], ref_ee_pos_[1], ref_ee_pos_[2], desired_j6_
        );
      }
    );

    // Sub Explicit J6 orientation target (rad) for Cartesian NMPC rotation without losing mode
    target_j6_sub_ = this->create_subscription<std_msgs::msg::Float64>(
      "/nmpc/target_j6", 10,
      [this](const std_msgs::msg::Float64::SharedPtr msg) {
        desired_j6_ = msg->data;
      }
    );

    // Sub Target joint pose (for MoveJ / HOME / Jogging)
    target_joint_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      "/nmpc/target_joint_pose", 10,
      [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
        if (msg->data.size() == 6) {
          target_joint_pos_ = msg->data;
          desired_j6_ = msg->data[5]; // Remember the user's commanded J6 gripper orientation!
          has_target_joint_ = true;
          RCLCPP_INFO_THROTTLE(
            this->get_logger(), *this->get_clock(), 1000,
            "New Target Joint Pose: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]",
            target_joint_pos_[0], target_joint_pos_[1], target_joint_pos_[2],
            target_joint_pos_[3], target_joint_pos_[4], target_joint_pos_[5]
          );
        }
      }
    );

    // Sub Obstacle position (legacy PointStamped)
    obstacle_sub_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
      "/nmpc/obstacle", 10,
      [this](const geometry_msgs::msg::PointStamped::SharedPtr msg) {
        obs_pos_[0] = msg->point.x;
        obs_pos_[1] = msg->point.y;
        obs_pos_[2] = msg->point.z;
        RCLCPP_INFO_THROTTLE(
          this->get_logger(), *this->get_clock(), 1000,
          "New Obstacle: [%.3f, %.3f, %.3f]",
          obs_pos_[0], obs_pos_[1], obs_pos_[2]
        );
      }
    );

    // Sub Full Obstacle Geometry from Vision/GUI/DynamicNode: [x, y, z_top, radius, (opt) vx, vy]
    obstacle_info_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      "/nmpc/obstacle_info", 10,
      [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
        if (msg->data.size() >= 4) {
          obs_pos_[0] = msg->data[0];
          obs_pos_[1] = msg->data[1];
          obs_z_top_ = std::clamp(msg->data[2], 0.08, 0.65);
          obs_pos_[2] = obs_z_top_ * 0.5;
          obs_radius_ = std::clamp(msg->data[3], 0.02, 0.25);
          if (msg->data.size() >= 6) {
            obs_vel_[0] = std::clamp(msg->data[4], -0.6, 0.6);
            obs_vel_[1] = std::clamp(msg->data[5], -0.6, 0.6);
            last_obs_vel_time_ = this->now();
          } else {
            obs_vel_[0] = 0.0;
            obs_vel_[1] = 0.0;
          }
          RCLCPP_INFO_THROTTLE(
            this->get_logger(), *this->get_clock(), 1000,
            "Synced Obstacle Info: [%.3f, %.3f] V=[%.2f, %.2f] H=%.2fm R=%.1fcm",
            obs_pos_[0], obs_pos_[1], obs_vel_[0], obs_vel_[1], obs_z_top_, obs_radius_ * 100.0
          );
        }
      }
    );

    // 3. Publishers
    cmd_pos_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
      "/dsr_position_controller/commands", 10
    );

    cmd_vel_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
      "/nmpc/joint_velocity_command", 10
    );

    current_ee_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
      "/nmpc/current_ee_pose", 10
    );

    cmd_traj_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
      "/dsr_controller2/joint_trajectory", 10
    );

    // 4. Timer loop running at 50 Hz (20 ms)
    control_timer_ = this->create_wall_timer(
      20ms, std::bind(&M0609NmpcNode::controlLoop, this)
    );

    RCLCPP_INFO(this->get_logger(), "M0609 NMPC Controller Node Ready (50 Hz)!");
  }

  ~M0609NmpcNode() override
  {
    if (capsule_) {
      ocp_m0609_kinematic_nmpc_41c0c9b9_acados_free(capsule_);
      ocp_m0609_kinematic_nmpc_41c0c9b9_acados_free_capsule(capsule_);
    }
  }

private:
  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    for (size_t i = 0; i < msg->name.size(); ++i) {
      for (int j = 0; j < 6; ++j) {
        std::string j_name = "joint_" + std::to_string(j + 1);
        if (msg->name[i] == j_name) {
          if (i < msg->position.size()) current_q_[j] = msg->position[i];
          if (i < msg->velocity.size()) current_dq_[j] = msg->velocity[i];
        }
      }
    }
    has_joint_state_ = true;
  }

  void controlLoop()
  {
    if (!has_joint_state_) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Waiting for joint_states..."
      );
      return;
    }

    double dt = 0.02; // 20 ms (50 Hz)

    // A. Check if we are in Joint Mode (MoveJ / HOME / Jogging)
    if (has_target_joint_) {
      double cur_ee[3];
      computeForwardKinematics(current_q_, cur_ee);
      double cur_obs_dist = std::hypot(cur_ee[0] - obs_pos_[0], cur_ee[1] - obs_pos_[1]);
      bool obs_intruding = (cur_ee[2] < obs_z_top_ + 0.11) && (cur_obs_dist < obs_radius_ + 0.175);

      if (obs_intruding) {
        // Obstacle approaching during Joint Move -> immediately hand off to active Cartesian NMPC so robot dodges!
        double tgt_ee[3];
        computeForwardKinematics(target_joint_pos_, tgt_ee);
        ref_ee_pos_[0] = tgt_ee[0];
        ref_ee_pos_[1] = tgt_ee[1];
        ref_ee_pos_[2] = tgt_ee[2];
        desired_j6_ = target_joint_pos_[5];
        has_target_joint_ = false;
        need_warmstart_reset_ = true;
      } else {
        auto pos_msg = std_msgs::msg::Float64MultiArray();
        pos_msg.data.resize(6);
        bool reached = true;
        double max_dq = 1.5 * dt; // max joint step ~0.03 rad (~1.7 deg per 20ms -> 85 deg/s)

        for (int i = 0; i < 6; ++i) {
          double diff = target_joint_pos_[i] - current_q_[i];
          if (std::abs(diff) > 0.008) reached = false;
          double step = std::clamp(diff, -max_dq, max_dq);
          pos_msg.data[i] = current_q_[i] + step;
        }
        cmd_pos_pub_->publish(pos_msg);

        ref_ee_pos_[0] = cur_ee[0];
        ref_ee_pos_[1] = cur_ee[1];
        ref_ee_pos_[2] = cur_ee[2];

        auto ee_msg = geometry_msgs::msg::PoseStamped();
        ee_msg.header.stamp = this->now();
        ee_msg.header.frame_id = "base_link";
        ee_msg.pose.position.x = cur_ee[0];
        ee_msg.pose.position.y = cur_ee[1];
        ee_msg.pose.position.z = cur_ee[2];
        ee_msg.pose.orientation.w = 1.0;
        current_ee_pub_->publish(ee_msg);

        // Once settled inside safe NMPC joint bounds, transition cleanly to 50Hz Cartesian NMPC hold
        // so the robot continuously avoids dynamic/rail obstacles even when standing still at HOME!
        bool in_nmpc_bounds =
          (target_joint_pos_[1] >= -0.18 && target_joint_pos_[1] <= 1.48) &&
          (target_joint_pos_[2] >= 0.27 && target_joint_pos_[2] <= 2.53) &&
          (target_joint_pos_[4] >= 0.27 && target_joint_pos_[4] <= 2.48);
        if (reached && in_nmpc_bounds) {
          double tgt_ee[3];
          computeForwardKinematics(target_joint_pos_, tgt_ee);
          ref_ee_pos_[0] = tgt_ee[0];
          ref_ee_pos_[1] = tgt_ee[1];
          ref_ee_pos_[2] = tgt_ee[2];
          desired_j6_ = target_joint_pos_[5];
          has_target_joint_ = false;
          need_warmstart_reset_ = true;
          RCLCPP_INFO(
            this->get_logger(),
            "Joint Pose reached -> handing over to 50Hz NMPC active guard at [%.3f, %.3f, %.3f]",
            ref_ee_pos_[0], ref_ee_pos_[1], ref_ee_pos_[2]
          );
        } else if (reached) {
          RCLCPP_INFO_THROTTLE(
            this->get_logger(), *this->get_clock(), 1000,
            "Target Joint Pose reached & held: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]",
            target_joint_pos_[0], target_joint_pos_[1], target_joint_pos_[2],
            target_joint_pos_[3], target_joint_pos_[4], target_joint_pos_[5]
          );
          return;
        } else {
          return;
        }
      }
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Reset acados trajectory warm-start cleanly when transitioning out of Joint Mode
    if (need_warmstart_reset_) {
      double u_zero[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      for (int stage = 0; stage <= OCP_M0609_KINEMATIC_NMPC_41C0C9B9_N; ++stage) {
        ocp_nlp_out_set(
          capsule_->nlp_config, capsule_->nlp_dims, capsule_->nlp_out, capsule_->nlp_in, stage, "x", current_q_.data()
        );
        if (stage < OCP_M0609_KINEMATIC_NMPC_41C0C9B9_N) {
          ocp_nlp_out_set(
            capsule_->nlp_config, capsule_->nlp_dims, capsule_->nlp_out, capsule_->nlp_in, stage, "u", u_zero
          );
        }
      }
      need_warmstart_reset_ = false;
    }

    // 1. Update initial state x0
    ocp_nlp_constraints_model_set(
      capsule_->nlp_config, capsule_->nlp_dims, capsule_->nlp_in, capsule_->nlp_out, 0, "lbx", current_q_.data()
    );
    ocp_nlp_constraints_model_set(
      capsule_->nlp_config, capsule_->nlp_dims, capsule_->nlp_in, capsule_->nlp_out, 0, "ubx", current_q_.data()
    );

    // 2. Prepare parameters: [obs_x, obs_y, obs_z_c, obs_radius, obs_z_top] with velocity prediction across horizon
    double vx = obs_vel_[0];
    double vy = obs_vel_[1];
    if ((this->now() - last_obs_vel_time_).seconds() > 0.5) {
      vx = 0.0;
      vy = 0.0;
    }
    double obs_speed = std::hypot(vx, vy);
    double r_dyn = obs_radius_ + std::min(0.035, 0.075 * obs_speed);

    double p_stage[5];
    p_stage[2] = obs_pos_[2];
    p_stage[3] = r_dyn;
    p_stage[4] = obs_z_top_;

    for (int stage = 0; stage <= OCP_M0609_KINEMATIC_NMPC_41C0C9B9_N; ++stage) {
      double t_pred = std::min(0.55, stage * 0.05);
      p_stage[0] = obs_pos_[0] + vx * t_pred;
      p_stage[1] = obs_pos_[1] + vy * t_pred;
      ocp_m0609_kinematic_nmpc_41c0c9b9_acados_update_params(capsule_, stage, p_stage, 5);
    }

    // 2b. Stationary Dodge + Anti-deadlock Guiding Waypoint (predicting obstacle 0.25s ahead)
    double obs_pred_x = obs_pos_[0] + vx * 0.25;
    double obs_pred_y = obs_pos_[1] + vy * 0.25;
    double cur_ee_now[3];
    computeForwardKinematics(current_q_, cur_ee_now);
    double active_ref[3] = {ref_ee_pos_[0], ref_ee_pos_[1], ref_ee_pos_[2]};
    double r_block = r_dyn + 0.140;
    double d_ref_obs = std::hypot(ref_ee_pos_[0] - obs_pred_x, ref_ee_pos_[1] - obs_pred_y);
    double d_ref_now = std::hypot(ref_ee_pos_[0] - obs_pos_[0], ref_ee_pos_[1] - obs_pos_[1]);

    // Case 1: Robot holding near ref_ee_pos_ inside obstacle Y-band (e.g. HOME y=0, z=0.236) while moving/rail obstacle approaches!
    // Lift cleanly UPWARD above obstacle top (ref_z < obs_z_top + 0.04) without snapping high parabolic arcs backward.
    double r_idle_dodge = r_dyn + 0.165;
    bool holding_near_ref = std::hypot(cur_ee_now[0] - ref_ee_pos_[0], cur_ee_now[1] - ref_ee_pos_[1]) < 0.08;
    if (holding_near_ref && std::abs(ref_ee_pos_[1] - obs_pred_y) < 0.14 && std::min(d_ref_obs, d_ref_now) < r_idle_dodge && ref_ee_pos_[2] < obs_z_top_ + 0.04) {
      active_ref[0] = std::max(0.28, std::min(ref_ee_pos_[0], obs_pred_x - 0.12));
      active_ref[1] = ref_ee_pos_[1];
      active_ref[2] = std::max(ref_ee_pos_[2], obs_z_top_ + 0.14);
    } else {
      // Case 2: Long cross-table move intersecting obstacle corridor
      double dx = ref_ee_pos_[0] - cur_ee_now[0];
      double dy = ref_ee_pos_[1] - cur_ee_now[1];
      double seg_len = std::hypot(dx, dy);
      // Only detour sideways on genuinely low cross-table moves below/near obstacle top (do not yank high arcs backward!)
      if (seg_len > 0.12 && (cur_ee_now[2] < obs_z_top_ + 0.045) && (ref_ee_pos_[2] < obs_z_top_ + 0.045)) {
        double t_proj = std::clamp(
          ((obs_pred_x - cur_ee_now[0]) * dx + (obs_pred_y - cur_ee_now[1]) * dy) / (seg_len * seg_len),
          0.0, 1.0
        );
        if (t_proj > 0.08 && t_proj < 0.92) {
          double qx = cur_ee_now[0] + t_proj * dx;
          double qy = cur_ee_now[1] + t_proj * dy;
          double d_seg = std::hypot(qx - obs_pred_x, qy - obs_pred_y);
          if (d_seg < r_block) {
            double dir_x = qx - obs_pred_x;
            double dir_y = qy - obs_pred_y;
            double nrm = std::hypot(dir_x, dir_y);
            if (nrm < 0.02) {
              dir_x = -1.0;
              dir_y = (cur_ee_now[1] >= 0.0) ? 0.25 : -0.25;
              nrm = std::hypot(dir_x, dir_y);
            }
            active_ref[0] = obs_pred_x + (dir_x / nrm) * (r_block + 0.04);
            active_ref[1] = obs_pred_y + (dir_y / nrm) * (r_block + 0.04);
            active_ref[2] = std::max(ref_ee_pos_[2], obs_z_top_ + 0.11);
          }
        }
      }
    }

    // 3. Compute nominal arm posture:
    double yaw_base = std::atan2(active_ref[1], active_ref[0]);
    double q_nom[6] = {yaw_base, 0.45, 1.45, 0.0, 1.25, desired_j6_};

    // yref: [p_tcp (3), z_tcp (3), q (6), u (6)] = 18 elements
    // z_tcp ref is [0, 0, -1] -> gripper pointing straight down!
    double yref[18];
    yref[0] = active_ref[0];
    yref[1] = active_ref[1];
    yref[2] = active_ref[2];
    yref[3] = 0.0;
    yref[4] = 0.0;
    yref[5] = -1.0;
    for (int i = 0; i < 6; ++i) yref[6 + i] = q_nom[i];
    for (int i = 0; i < 6; ++i) yref[12 + i] = 0.0;

    // yref_e: [p_tcp (3), z_tcp (3), q (6)] = 12 elements
    double yref_e[12];
    yref_e[0] = active_ref[0];
    yref_e[1] = active_ref[1];
    yref_e[2] = active_ref[2];
    yref_e[3] = 0.0;
    yref_e[4] = 0.0;
    yref_e[5] = -1.0;
    for (int i = 0; i < 6; ++i) yref_e[6 + i] = q_nom[i];

    for (int stage = 0; stage < OCP_M0609_KINEMATIC_NMPC_41C0C9B9_N; ++stage) {
      ocp_nlp_cost_model_set(
        capsule_->nlp_config, capsule_->nlp_dims, capsule_->nlp_in, stage, "yref", yref
      );
    }
    ocp_nlp_cost_model_set(
      capsule_->nlp_config, capsule_->nlp_dims, capsule_->nlp_in,
      OCP_M0609_KINEMATIC_NMPC_41C0C9B9_N, "yref", yref_e
    );

    // 4. Solve NMPC via RTI
    int status = ocp_m0609_kinematic_nmpc_41c0c9b9_acados_solve(capsule_);
    auto end_time = std::chrono::high_resolution_clock::now();
    double solve_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    if (status != 0 && status != 2) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 1000,
        "acados solve returned non-zero status: %d (time: %.2f ms)", status, solve_time_ms
      );
      return;
    }

    // 5. Extract optimal control action u0 = dq*
    double u0[6];
    ocp_nlp_out_get(capsule_->nlp_config, capsule_->nlp_dims, capsule_->nlp_out, 0, "u", u0);

    // Publish velocity command
    auto vel_msg = std_msgs::msg::Float64MultiArray();
    vel_msg.data.resize(6);
    for (int i = 0; i < 6; ++i) {
      vel_msg.data[i] = u0[i];
    }
    cmd_vel_pub_->publish(vel_msg);

    // Publish position command for Gazebo ForwardCommandController with safety step limiter
    auto pos_msg = std_msgs::msg::Float64MultiArray();
    pos_msg.data.resize(6);
    for (int i = 0; i < 6; ++i) {
      // Clamp max change to 0.045 rad (~2.6 deg) per 20ms step -> brisk, responsive tracking (up to 130 deg/s)
      double dq_step = std::clamp(u0[i] * dt, -0.045, 0.045);
      pos_msg.data[i] = current_q_[i] + dq_step;
    }
    cmd_pos_pub_->publish(pos_msg);

    // Compute and publish current EE position via FK
    double cur_ee[3];
    computeForwardKinematics(current_q_, cur_ee);
    auto ee_msg = geometry_msgs::msg::PoseStamped();
    ee_msg.header.stamp = this->now();
    ee_msg.header.frame_id = "base_link";
    ee_msg.pose.position.x = cur_ee[0];
    ee_msg.pose.position.y = cur_ee[1];
    ee_msg.pose.position.z = cur_ee[2];
    ee_msg.pose.orientation.w = 1.0;
    current_ee_pub_->publish(ee_msg);

    // Optional trajectory command
    trajectory_msgs::msg::JointTrajectory traj_msg;
    traj_msg.header.stamp = this->now();
    traj_msg.joint_names = {
      "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"
    };
    trajectory_msgs::msg::JointTrajectoryPoint pt;
    pt.positions = pos_msg.data;
    pt.velocities.resize(6);
    for (int i = 0; i < 6; ++i) {
      pt.velocities[i] = u0[i];
    }
    pt.time_from_start = rclcpp::Duration::from_seconds(dt);
    traj_msg.points.push_back(pt);
    cmd_traj_pub_->publish(traj_msg);

    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 1000,
      "NMPC (%.1f ms) | EE: [%.3f, %.3f, %.3f] -> Ref: [%.3f, %.3f, %.3f]",
      solve_time_ms, cur_ee[0], cur_ee[1], cur_ee[2],
      ref_ee_pos_[0], ref_ee_pos_[1], ref_ee_pos_[2]
    );
  }

  void computeForwardKinematics(const std::vector<double>& q, double ee_pos[3])
  {
    double p_dummy[5] = {0};
    double res_buf[12] = {0};
    const double* arg[2] = { q.data(), p_dummy };
    double* res[1] = { res_buf };
    m0609_kinematic_nmpc_cost_y_e_fun(arg, res, nullptr, nullptr, nullptr);
    ee_pos[0] = res_buf[0];
    ee_pos[1] = res_buf[1];
    ee_pos[2] = res_buf[2];
  }

  ocp_m0609_kinematic_nmpc_41c0c9b9_solver_capsule* capsule_{nullptr};

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_fallback_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr target_j6_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr target_joint_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr obstacle_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr obstacle_info_sub_;

  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr cmd_pos_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr current_ee_pub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr cmd_traj_pub_;
  rclcpp::TimerBase::SharedPtr control_timer_;

  std::vector<double> current_q_;
  std::vector<double> current_dq_;
  std::vector<double> ref_ee_pos_;
  std::vector<double> target_joint_pos_;
  double desired_j6_;
  std::vector<double> obs_pos_;
  std::vector<double> obs_vel_;
  double obs_radius_;
  double obs_z_top_;
  bool has_joint_state_;
  bool has_target_joint_;
  bool need_warmstart_reset_;
  rclcpp::Time last_obs_vel_time_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<M0609NmpcNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
