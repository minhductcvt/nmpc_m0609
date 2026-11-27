import os
import numpy as np
import casadi as cs
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

def rpy_to_rot(rpy):
    r, p, y = rpy[0], rpy[1], rpy[2]
    cr, sr = cs.cos(r), cs.sin(r)
    cp, sp = cs.cos(p), cs.sin(p)
    cy, sy = cs.cos(y), cs.sin(y)
    Rz = cs.vertcat(
        cs.horzcat(cy, -sy, 0),
        cs.horzcat(sy, cy, 0),
        cs.horzcat(0, 0, 1)
    )
    Ry = cs.vertcat(
        cs.horzcat(cp, 0, sp),
        cs.horzcat(0, 1, 0),
        cs.horzcat(-sp, 0, cp)
    )
    Rx = cs.vertcat(
        cs.horzcat(1, 0, 0),
        cs.horzcat(0, cr, -sr),
        cs.horzcat(0, sr, cr)
    )
    return Rz @ Ry @ Rx

def rotz(angle):
    c, s = cs.cos(angle), cs.sin(angle)
    return cs.vertcat(
        cs.horzcat(c, -s, 0),
        cs.horzcat(s, c, 0),
        cs.horzcat(0, 0, 1)
    )

def hom(R, p):
    return cs.vertcat(
        cs.horzcat(R, p),
        cs.horzcat(0, 0, 0, 1)
    )

def forward_kinematics(q):
    origins = [
        ([0, 0, 0.1345], [0, 0, 0]),
        ([0, 0.0062, 0], [0, -1.571, -1.571]),
        ([0.411, 0, 0], [0, 0, 1.571]),
        ([0, -0.368, 0], [1.571, 0, 0]),
        ([0, 0, 0], [-1.571, 0, 0]),
        ([0, -0.121, 0], [1.571, 0, 0]),
    ]
    
    T = cs.SX.eye(4)
    positions = [] # store link positions
    
    for i in range(6):
        p_orig, rpy_orig = origins[i]
        T_joint_orig = hom(rpy_to_rot(rpy_orig), cs.SX(p_orig))
        T_rot = hom(rotz(q[i]), cs.SX.zeros(3, 1))
        T = T @ T_joint_orig @ T_rot
        positions.append(T[0:3, 3])
        
    # TCP frame (0.1884m = 0.0256m Doosan SVM + 0.1628m Robotiq 2F-85 along Z from link_6)
    T_tcp = T @ hom(cs.SX.eye(3), cs.SX([0, 0, 0.1884]))
    p_tcp = T_tcp[0:3, 3]
    R_tcp = T_tcp[0:3, 0:3]
    positions.append(p_tcp)
    
    return p_tcp, R_tcp, positions

def create_kinematic_nmpc_model():
    model_name = 'm0609_kinematic_nmpc'
    
    # State: Joint positions q in R^6
    q = cs.SX.sym('q', 6)
    q_dot = cs.SX.sym('q_dot', 6)
    
    # Control input: Joint velocities u = v in R^6
    u = cs.SX.sym('u', 6)
    
    # Online Parameters p: [obs_x, obs_y, obs_z_center, obs_radius, obs_z_top] -> 5 parameters
    p = cs.SX.sym('p', 5)
    p_obs = p[0:3]
    r_obs = p[3]
    z_obs_top = p[4]

    # Forward Kinematics
    p_tcp, R_tcp, link_positions = forward_kinematics(q)

    # Tool pointing direction (Z-axis of TCP): pointing straight down [0, 0, -1]
    z_tcp = R_tcp[0:3, 2]
    # Finger opening axis (X-axis of link_6 / Robotiq base in URDF)
    x_tcp = R_tcp[0:3, 0]

    # Explicit dynamics: q_dot = u
    f_expl = u
    f_impl = q_dot - u

    model = AcadosModel()
    model.name = model_name
    model.x = q
    model.xdot = q_dot
    model.u = u
    model.p = p
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl

    # Full-body + Robotiq 2F-85 Gripper Collision Control Points (8 points)
    p_elbow    = link_positions[2]
    p_forearm  = (link_positions[2] + link_positions[4]) * 0.5
    p_wrist    = link_positions[4]
    p_flange   = link_positions[5]
    p_palm     = (link_positions[5] + p_tcp) * 0.5
    p_finger_l = p_tcp + 0.055 * x_tcp - 0.025 * z_tcp
    p_finger_r = p_tcp - 0.055 * x_tcp - 0.025 * z_tcp
    p_ee       = p_tcp

    ctrl_pts = [
        (p_elbow,    0.080),
        (p_forearm,  0.075),
        (p_wrist,    0.085),
        (p_flange,   0.095),
        (p_palm,     0.115),
        (p_finger_l, 0.075),
        (p_finger_r, 0.075),
        (p_ee,       0.095),
    ]

    # Continuous vertical cylinder segment from z=0.04m up to z_obs_top (no gaps!)
    eps = 1e-4
    obs_dists = []
    for pt, r_safe in ctrl_pts:
        z_clamped = cs.fmin(cs.fmax(pt[2], 0.04), z_obs_top)
        p_closest = cs.vertcat(p_obs[0], p_obs[1], z_clamped)
        dist_m = cs.sqrt(cs.sumsqr(pt - p_closest) + eps)
        obs_dists.append(dist_m - (r_obs + r_safe))

    # Table floor safety clearance (prevent gripper from plunging into workbench/trays)
    floor_clear_tcp  = p_ee[2] - 0.015
    floor_clear_palm = p_palm[2] - 0.065

    h_expr = cs.vertcat(*obs_dists, floor_clear_tcp, floor_clear_palm)
    model.con_h_expr = h_expr
    model.con_h_expr_e = h_expr
    
    # Cost terms:
    # 1. Position tracking: p_tcp (3)
    # 2. Tool orientation tracking: z_tcp (3) -> ref [0, 0, -1]
    # 3. Posture regularization: q (6) -> keeps arm in natural elbow-up stance, no wrist twist
    # 4. Control effort: u (6)
    model.cost_y_expr = cs.vertcat(p_tcp, z_tcp, q, u)
    model.cost_y_expr_e = cs.vertcat(p_tcp, z_tcp, q)
    
    return model

def generate_kinematic_ocp_solver():
    model = create_kinematic_nmpc_model()
    ocp = AcadosOcp()
    ocp.model = model
    
    # Prediction Horizon: 1.25s lookahead for smoother detour planning
    N = 25
    Tf = 1.25 # dt = 0.05s
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf
    
    # Cost formulation: NONLINEAR_LS
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    
    # Weights (high Cartesian accuracy across wide 110x40cm cell + conveyor while keeping natural elbow-up bias)
    W_pos = 1200.0 * np.eye(3)  # sub-5mm steady-state EE position tracking
    W_ori = 120.0 * np.eye(3)   # force gripper pointing straight down
    W_q   = np.diag([0.2, 0.6, 0.8, 1.5, 0.8, 0.8])  # gentle elbow-up regularization without pulling EE inward
    W_vel = 0.08 * np.eye(6)    # smooth joint velocities
    
    W = np.block([
        [W_pos, np.zeros((3, 3)), np.zeros((3, 6)), np.zeros((3, 6))],
        [np.zeros((3, 3)), W_ori, np.zeros((3, 6)), np.zeros((3, 6))],
        [np.zeros((6, 3)), np.zeros((6, 3)), W_q, np.zeros((6, 6))],
        [np.zeros((6, 3)), np.zeros((6, 3)), np.zeros((6, 6)), W_vel]
    ])
    
    W_pos_e = 2400.0 * np.eye(3)
    W_ori_e = 240.0 * np.eye(3)
    W_q_e   = np.diag([0.4, 1.2, 1.6, 3.0, 1.6, 1.6])
    
    W_e = np.block([
        [W_pos_e, np.zeros((3, 3)), np.zeros((3, 6))],
        [np.zeros((3, 3)), W_ori_e, np.zeros((3, 6))],
        [np.zeros((6, 3)), np.zeros((6, 3)), W_q_e]
    ])
    
    ocp.cost.W = W
    ocp.cost.W_e = W_e
    
    # yref dimension: 3 + 3 + 6 + 6 = 18
    ocp.cost.yref = np.zeros(18)
    # yref_e dimension: 3 + 3 + 6 = 12
    ocp.cost.yref_e = np.zeros(12)
    
    # Natural joint operating limits (strictly prevents elbow inversion and backward wrist twist)
    q_min = np.array([-3.14, -0.20,  0.25, -1.20,  0.25, -3.14])
    q_max = np.array([ 3.14,  1.50,  2.55,  1.20,  2.50,  3.14])
    
    # Velocity limits (rad/s)
    v_max = np.array([2.5, 2.5, 3.0, 3.5, 3.5, 3.5])
    v_min = -v_max
    
    # State box constraints on q
    ocp.constraints.lbx = q_min
    ocp.constraints.ubx = q_max
    ocp.constraints.idxbx = np.arange(6)
    
    ocp.constraints.lbx_e = q_min
    ocp.constraints.ubx_e = q_max
    ocp.constraints.idxbx_e = np.arange(6)
    
    # Input box constraints on u = v
    ocp.constraints.lbu = v_min
    ocp.constraints.ubu = v_max
    ocp.constraints.idxbu = np.arange(6)
    
    # 10 Euclidean distance constraints (8 obstacle capsule + 2 floor): h >= 0
    n_h = 10
    ocp.constraints.lh = np.zeros(n_h)
    ocp.constraints.uh = 1e9 * np.ones(n_h)
    ocp.constraints.lh_e = np.zeros(n_h)
    ocp.constraints.uh_e = 1e9 * np.ones(n_h)
    
    # Slack variables on h to guarantee 100% QP feasibility at all times
    ocp.constraints.Jsh = np.eye(n_h)
    ocp.constraints.Jsh_e = np.eye(n_h)
    
    # Strong Euclidean slack penalties (repel strongly rather than cutting through obstacle!)
    ocp.cost.zl = 4000.0 * np.ones(n_h)
    ocp.cost.Zl = 40000.0 * np.ones(n_h)
    ocp.cost.zu = np.zeros(n_h)
    ocp.cost.Zu = np.zeros(n_h)
    
    ocp.cost.zl_e = 4000.0 * np.ones(n_h)
    ocp.cost.Zl_e = 40000.0 * np.ones(n_h)
    ocp.cost.zu_e = np.zeros(n_h)
    ocp.cost.Zu_e = np.zeros(n_h)
    
    # Initial state matching HOME_SAFE posture and parameters [obs_x, obs_y, obs_z_c, obs_r, obs_z_top]
    ocp.constraints.x0 = np.array([0.0, 0.0, 1.5708, 0.0, 1.5708, 0.0])
    p_init = np.array([0.45, 0.0, 0.15, 0.05, 0.25])
    ocp.parameter_values = p_init
    
    # Solver options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.qp_solver_cond_N = 5
    
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    export_dir = os.path.join(pkg_root, 'c_generated_solver_kinematic')
    ocp.code_export_directory = export_dir
    
    print(f'Generating C-code OCP solver in: {export_dir} ...')
    json_file = os.path.join(export_dir, 'acados_ocp.json')
    solver = AcadosOcpSolver(ocp, json_file=json_file)
    print('SUCCESS! C-code OCP Solver generated successfully!')
    return solver

if __name__ == '__main__':
    generate_kinematic_ocp_solver()
