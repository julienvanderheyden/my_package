#!/usr/bin/env python3
"""
shadow_hand_simulation.py
─────────────────────────
ROS-based digital twin for the Shadow Hand using MuJoCo.

ROS interface
─────────────
  PUBLISH  /shadowhand_state_topic   sensor_msgs/JointState   24 hand joints
  SUBSCRIBE /shadowhand_command_topic std_msgs/Float64MultiArray   24 joint targets

Joint ordering
──────────────
  Published joints  : JOINT_NAMES  (24 entries, read from qpos)
  Subscribed commands: 24 values in JOINT_NAMES order (written to data.ctrl)

Coupling mode (auto-detected at startup from the loaded MJCF)
──────────────────────────────────────────────────────────────
  COUPLED   (standard model, e.g. scene_right_perso.xml)
    rh_A_FFJ0/MFJ0/RFJ0/LFJ0 each drive J2+J1 via tendon.
    The subscriber sums J2_target + J1_target before writing to the J0 actuator.
    Total actuators: 20 (16 direct + 4 coupled J0).

  UNCOUPLED (perso model, e.g. scene_right_perso_uncoupled.xml)
    Every joint has its own actuator (rh_A_FFJ1, rh_A_FFJ2, …).
    Commands map 1-to-1; no summation is performed.
    Total actuators: 24 (one per joint, no J0).
"""

import ctypes
import os
import threading
import time

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Empty
import rospkg

# ── Windows high-resolution timer (no-op on Linux) ────────────────────────────
try:
    ctypes.windll.winmm.timeBeginPeriod(1)
    _WINDOWS_TIMER = True
except AttributeError:
    _WINDOWS_TIMER = False


def precise_sleep(duration: float) -> None:
    """Busy-wait sleep for accurate real-time pacing on Windows."""
    end = time.perf_counter() + duration
    sleep_until = end - 0.001
    if sleep_until > time.perf_counter():
        time.sleep(sleep_until - time.perf_counter())
    while time.perf_counter() < end:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# 1. JOINT / ACTUATOR MAPS
# ══════════════════════════════════════════════════════════════════════════════

# 24 joints published on /shadowhand_state_topic and received on
# /shadowhand_command_topic — both topics share this exact ordering.
JOINT_NAMES = [
    "rh_WRJ1", "rh_WRJ2",
    "rh_FFJ1", "rh_FFJ2", "rh_FFJ3", "rh_FFJ4",
    "rh_MFJ1", "rh_MFJ2", "rh_MFJ3", "rh_MFJ4",
    "rh_RFJ1", "rh_RFJ2", "rh_RFJ3", "rh_RFJ4",
    "rh_LFJ1", "rh_LFJ2", "rh_LFJ3", "rh_LFJ4", "rh_LFJ5",
    "rh_THJ1", "rh_THJ2", "rh_THJ3", "rh_THJ4", "rh_THJ5",
]

# ── Coupled model (standard): 16 direct + 4 J0 tendon actuators ───────────────
# Direct actuators (1 joint → 1 actuator):
DIRECT_ACTUATOR_NAMES = [
    "rh_A_WRJ2", "rh_A_WRJ1",
    "rh_A_FFJ4", "rh_A_FFJ3",
    "rh_A_MFJ4", "rh_A_MFJ3",
    "rh_A_RFJ4", "rh_A_RFJ3",
    "rh_A_LFJ5", "rh_A_LFJ4", "rh_A_LFJ3",
    "rh_A_THJ5", "rh_A_THJ4", "rh_A_THJ3", "rh_A_THJ2", "rh_A_THJ1",
]

# Coupled actuators: one actuator drives two joints via tendon (J2 + J1).
# The subscriber receives individual J1/J2 targets and sums them before
# writing to the single J0 ctrl channel.
COUPLED_ACTUATOR_NAMES = [
    "rh_A_FFJ0",   # drives rh_FFJ2 + rh_FFJ1
    "rh_A_MFJ0",   # drives rh_MFJ2 + rh_MFJ1
    "rh_A_RFJ0",   # drives rh_RFJ2 + rh_RFJ1
    "rh_A_LFJ0",   # drives rh_LFJ2 + rh_LFJ1
]

ALL_ACTUATOR_NAMES_COUPLED = DIRECT_ACTUATOR_NAMES + COUPLED_ACTUATOR_NAMES + ["rh_A_arm_lift"]  # 21 total

# Coupled joints map: J0 actuator → (J2 joint, J1 joint)
# Used both in apply_initial_config and _command_callback (coupled mode).
COUPLED_JOINTS = {
    "rh_A_FFJ0": ("rh_FFJ2", "rh_FFJ1"),
    "rh_A_MFJ0": ("rh_MFJ2", "rh_MFJ1"),
    "rh_A_RFJ0": ("rh_RFJ2", "rh_RFJ1"),
    "rh_A_LFJ0": ("rh_LFJ2", "rh_LFJ1"),
}

# ── Uncoupled model (perso): one actuator per joint, no J0 ────────────────────
# Actuator names mirror JOINT_NAMES exactly ("rh_" → "rh_A_").
ALL_ACTUATOR_NAMES_UNCOUPLED = (
    [name.replace("rh_", "rh_A_") for name in JOINT_NAMES]
    + ["rh_A_arm_lift"]
)   # 25 total


def detect_coupling_mode(model) -> bool:
    """
    Return True  → coupled model   (standard hand, J0 tendon actuators present).
    Return False → uncoupled model (perso hand, one actuator per joint).

    Detection probes for rh_A_FFJ0: it exists in the coupled model and is
    absent in the uncoupled one.
    """
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rh_A_FFJ0") != -1

# Initial hand pose (joint_name → angle in rad)
# MEDIUM WRAP PRESHAPE
INITIAL_CONFIG = {
    "rh_FFJ4": 0.0, "rh_FFJ3": 0.0, "rh_FFJ2": 0.0, "rh_FFJ1": 0.0,
    "rh_MFJ4": 0.0, "rh_MFJ3": 0.0, "rh_MFJ2": 0.0, "rh_MFJ1": 0.0,
    "rh_RFJ4": 0.0, "rh_RFJ3": 0.0, "rh_RFJ2": 0.0, "rh_RFJ1": 0.0,
    "rh_LFJ5": 0.0, "rh_LFJ4": 0.0, "rh_LFJ3": 0.0, "rh_LFJ2": 0.0, "rh_LFJ1": 0.0,
    "rh_THJ5": 0.0, "rh_THJ4": 1.21, "rh_THJ3": 0.0, "rh_THJ2": 0.0, "rh_THJ1": 0.0,
    "rh_WRJ2": 0.0, "rh_WRJ1": 0.0,
    "rh_arm_lift": 0.02,   # 2 cm above ground — avoids contact force at rest
}

# LATERAL PINCH PRESHAPE
# INITIAL_CONFIG = {
#     "rh_FFJ4": 0.0, "rh_FFJ3": 1.0, "rh_FFJ2": 1.57, "rh_FFJ1": 0.0,
#     "rh_MFJ4": 0.0, "rh_MFJ3": 1.57, "rh_MFJ2": 1.57, "rh_MFJ1": 1.57,
#     "rh_RFJ4": 0.0, "rh_RFJ3": 1.57, "rh_RFJ2": 1.57, "rh_RFJ1": 1.57,
#     "rh_LFJ5": 0.0, "rh_LFJ4": 0.0, "rh_LFJ3": 1.57, "rh_LFJ2": 1.57, "rh_LFJ1": 1.57,
#     "rh_THJ5": 0.0, "rh_THJ4": 0.0, "rh_THJ3": 0.0, "rh_THJ2": -0.7, "rh_THJ1": 0.0,
#     "rh_WRJ2": 0.0, "rh_WRJ1": 0.0, "rh_arm_lift" : 0.1
# }


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_index_maps(model, actuator_names: list):
    """
    Pre-compute integer indices so the hot loop never calls mj_name2id.

    Parameters
    ----------
    model          : MjModel
    actuator_names : list[str]
        Pass ALL_ACTUATOR_NAMES_COUPLED or ALL_ACTUATOR_NAMES_UNCOUPLED
        depending on the model loaded (see detect_coupling_mode).

    Returns
    -------
    joint_qpos_ids : list[int]       qpos address for each entry in JOINT_NAMES
    joint_qvel_ids : list[int]       dof address for each entry in JOINT_NAMES
    actuator_ids   : dict[str, int]  actuator name → ctrl index
    """
    joint_qpos_ids = []
    joint_qvel_ids = []
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid == -1:
            raise RuntimeError(f"Joint '{name}' not found in model.")
        joint_qpos_ids.append(model.jnt_qposadr[jid])
        joint_qvel_ids.append(model.jnt_dofadr[jid])

    actuator_ids = {}
    for name in actuator_names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid == -1:
            raise RuntimeError(f"Actuator '{name}' not found in model.")
        actuator_ids[name] = aid

    return joint_qpos_ids, joint_qvel_ids, actuator_ids


def apply_initial_config(model, data, config: dict, actuator_ids: dict, coupled: bool):
    """
    Set qpos and ctrl so the hand starts in `config` without fighting itself.

    Parameters
    ----------
    coupled : bool
        True  → J0 tendon model: J0 ctrl = J2_angle + J1_angle.
        False → uncoupled model: every joint maps 1-to-1 to its actuator.
    """
    # 1. All joint positions
    for joint_name, angle in config.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid != -1:
            data.qpos[model.jnt_qposadr[jid]] = angle

    if coupled:
        # 2a. Direct actuators (joint_name → rh_A_<joint_name>)
        for joint_name, angle in config.items():
            actuator_name = joint_name.replace("rh_", "rh_A_")
            if actuator_name in actuator_ids:
                data.ctrl[actuator_ids[actuator_name]] = angle

        # 2b. Coupled J0 actuators: ctrl = J2_angle + J1_angle
        for actuator_name, (j2_name, j1_name) in COUPLED_JOINTS.items():
            if actuator_name in actuator_ids:
                data.ctrl[actuator_ids[actuator_name]] = (
                    config.get(j2_name, 0.0) + config.get(j1_name, 0.0)
                )
    else:
        # 2c. Uncoupled: every joint maps directly to rh_A_<joint>
        for joint_name, angle in config.items():
            actuator_name = joint_name.replace("rh_", "rh_A_")
            if actuator_name in actuator_ids:
                data.ctrl[actuator_ids[actuator_name]] = angle

    mujoco.mj_forward(model, data)


# ══════════════════════════════════════════════════════════════════════════════
# 3. GRASP LOGGER
# ══════════════════════════════════════════════════════════════════════════════

class GraspLogger:
    """
    Accumulates simulation data every physics step and produces a
    post-simulation plot when plot() is called.

    Logged quantities
    ─────────────────
      time         : simulation time (s)
      joint_torques: qfrc_actuator for all 24 hand joints  (Nm)
      n_contacts   : number of active contact pairs
      normal_force : total normal force summed over all contacts (N)
      shear_force  : total tangential force magnitude summed over all contacts (N)
      cop_x/y/z    : centre of pressure in world frame (m)
      pen_depth    : maximum penetration depth (m, negative = penetrating)
      grasp_quality: friction-cone margin proxy — min(μ·Fn - Ft) over contacts
      arm_lift_pos : arm_lift joint position (m)
    """

    # Joints to plot individually in the torque panel
    TORQUE_JOINTS = [
        "rh_FFJ3", "rh_FFJ2", "rh_MFJ3", "rh_MFJ2",
        "rh_RFJ3", "rh_RFJ2", "rh_LFJ3", "rh_LFJ2",
        "rh_THJ2", "rh_THJ1",
    ]

    def __init__(self, model, data, joint_names, joint_qpos_ids, actuator_ids):
        self._model          = model
        self._data           = data
        self._joint_names    = joint_names
        self._joint_qpos_ids = joint_qpos_ids
        self._actuator_ids   = actuator_ids

        # Pre-compute dof addresses for torque joints
        self._torque_dof_ids = []
        for name in self.TORQUE_JOINTS:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._torque_dof_ids.append(model.jnt_dofadr[jid])

        # Friction coefficient (read from first geom or use a default)
        self._mu = float(model.geom_friction[0, 0]) if model.ngeom > 0 else 0.5

        # Pre-allocate contact force buffer (mj_contactForce needs 6 floats)
        self._efc_buf = np.zeros(6)

        # History lists — appended every step
        self.time          = []
        self.joint_torques = []   # shape (T, len(TORQUE_JOINTS))
        self.n_contacts    = []
        self.normal_force  = []
        self.shear_force   = []
        self.pen_depth     = []

    def record(self):
        """Call once per physics step, after mj_step."""
        data  = self._data
        model = self._model

        self.time.append(data.time)

        # ── Joint torques (actuator contribution to generalised forces) ────────
        self.joint_torques.append(
            [data.qfrc_actuator[i] for i in self._torque_dof_ids]
        )
        # ── Contact metrics ────────────────────────────────────────────────────
        n   = data.ncon
        self.n_contacts.append(n)

        total_normal  = 0.0
        total_shear   = 0.0
        max_pen       = 0.0           # most negative dist value

        for j in range(n):
            contact = data.contact[j]
            mujoco.mj_contactForce(model, data, j, self._efc_buf)

            # efc_buf[0]   = normal force (positive = compressive in contact frame)
            # efc_buf[1:3] = tangential (shear) forces
            fn = float(self._efc_buf[0])
            ft = float(np.linalg.norm(self._efc_buf[1:3]))

            total_normal += fn
            total_shear  += ft

            # Penetration depth (contact.dist < 0 when penetrating)
            max_pen = min(max_pen, float(contact.dist))

        self.normal_force.append(total_normal)
        self.shear_force.append(total_shear)
        self.pen_depth.append(max_pen)

    def reset(self):
        """Clear history on GUI reset."""
        self.time.clear()
        self.joint_torques.clear()
        self.n_contacts.clear()
        self.normal_force.clear()
        self.shear_force.clear()
        self.pen_depth.clear()

    def plot(self):
        """Generate and display all post-simulation plots."""
        if not self.time:
            rospy.loginfo("No data recorded — skipping plots.")
            return

        t  = np.array(self.time)
        tq = np.array(self.joint_torques)   # (T, J)
        nf = np.array(self.normal_force)
        sf = np.array(self.shear_force)
        nc = np.array(self.n_contacts)
        pd = np.array(self.pen_depth) * 1e3 # convert to mm

        fig = plt.figure(figsize=(16, 18))
        fig.suptitle("Shadow Hand Grasp Analysis", fontsize=15, fontweight="bold")
        gs  = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35)

        # ── Panel 1: Joint torques ─────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        for i, name in enumerate(self.TORQUE_JOINTS):
            ax1.plot(t, tq[:, i], label=name, linewidth=0.9)
        ax1.set_title("Actuator Joint Torques")
        ax1.set_ylabel("Torque (Nm)")
        ax1.legend(ncol=5, fontsize=7, loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color="k", linewidth=0.5)

        # ── Panel 2: Normal vs shear force ─────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(t, nf, label="Normal (N)",  color="tab:blue")
        ax2.plot(t, sf, label="Shear (N)",   color="tab:orange")
        ax2.set_title("Contact Forces (summed over all contacts)")
        ax2.set_ylabel("Force (N)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # ── Panel 3: Number of contacts ────────────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(t, nc, color="tab:purple", linewidth=0.9)
        ax4.set_title("Active Contact Points")
        ax4.set_ylabel("Count")
        ax4.grid(True, alpha=0.3)

        # ── Panel 4: Penetration depth ─────────────────────────────────────────
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(t, pd, color="tab:red", linewidth=0.9)
        ax5.axhline(0, color="k", linewidth=0.5)
        ax5.set_title("Max Penetration Depth")
        ax5.set_ylabel("Depth (mm)")
        ax5.grid(True, alpha=0.3)

        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 4. ROS NODE
# ══════════════════════════════════════════════════════════════════════════════

class ShadowHandDigitalTwin:
    """
    Wraps the MuJoCo simulation and handles all ROS communication.

    Threading model
    ───────────────
      Main thread   : physics loop (500 Hz, real-time paced)
      viewer_thread : MuJoCo passive viewer sync (60 Hz)
      ROS spin      : rospy.spin() in a daemon thread; callbacks write
                      to self._ctrl_buffer under a lock
    """

    def __init__(self):
        # ── ROS init ──────────────────────────────────────────────────────────
        rospy.init_node("shadow_hand_digital_twin", anonymous=False)
        self._rate_hz   = 500           # must match 1 / model.opt.timestep
        self._ros_rate  = rospy.Rate(self._rate_hz)

        # Publisher: joint states at 500 Hz
        self._state_pub = rospy.Publisher(
            "/shadowhand_state_topic",
            JointState,
            queue_size=1,       # drop old messages; latency matters more than completeness
        )

        # Subscriber: actuator commands
        # The callback only updates a shared buffer; the physics loop reads it.
        self._ctrl_lock   = threading.Lock()
        self._ctrl_buffer = None        # None until first command arrives
        self._cmd_sub = rospy.Subscriber(
            "/shadowhand_command_topic",
            Float64MultiArray,
            self._command_callback,
            queue_size=1,
        )

        # Subscriber: lift trigger
        # Receiving any Empty message on /lift starts the lifting motion.
        # The physics loop reads _lift_active and ramps _lift_current smoothly.
        self._lift_active  = False
        self._lift_target  = 0.2    # target height (m) — tune to your scene
        self._lift_current = INITIAL_CONFIG["rh_arm_lift"]    # current ctrl setpoint, incremented each step
        self._lift_speed   = 0.05   # ramp rate (m/s) — tune for desired smoothness
        self._lift_sub = rospy.Subscriber(
            "/lift",
            Empty,
            self._lift_callback,
            queue_size=1,
        )

        # ── MuJoCo model ──────────────────────────────────────────────────────
        _ROS_PACK = rospkg.RosPack()
        _MODEL_PATH = os.path.join(
            _ROS_PACK.get_path("my_package"),
            "mjcf", "shadow_hand", "scene_right_perso.xml"
        )
        # _MODEL_PATH = os.path.join(
        #     _ROS_PACK.get_path("my_package"),
        #     "mjcf", "shadow_hand", "scene_lateral_pinch.xml"
        # )

        self._model = mujoco.MjModel.from_xml_path(_MODEL_PATH)
        self._data  = mujoco.MjData(self._model)

        # Auto-detect coupling mode from the loaded model
        self._coupled = detect_coupling_mode(self._model)
        _mode_label   = "coupled (J0 tendon)" if self._coupled else "uncoupled (per-joint)"
        rospy.loginfo("Coupling mode detected: %s", _mode_label)

        # Pre-compute index maps using the actuator list that matches the model
        _actuator_names = (
            ALL_ACTUATOR_NAMES_COUPLED if self._coupled else ALL_ACTUATOR_NAMES_UNCOUPLED
        )
        self._joint_qpos_ids, self._joint_qvel_ids, self._actuator_ids = (
            build_index_maps(self._model, _actuator_names)
        )

        # Grasp logger — records every physics step, plots on exit
        self._logger = GraspLogger(
            self._model, self._data,
            JOINT_NAMES, self._joint_qpos_ids, self._actuator_ids,
        )

        # ── Viewer ────────────────────────────────────────────────────────────
        # _viewer_ready is set by the viewer thread once launch_passive has
        # completed its internal mj_forward, so our apply_initial_config is
        # guaranteed to run after and not be overwritten.
        self._stop_event   = threading.Event()
        self._viewer_ready = threading.Event()
        self._viewer_thread = threading.Thread(
            target=self._run_viewer, daemon=True
        )
        self._viewer_thread.start()

        # ── ROS spin thread ───────────────────────────────────────────────────
        # Keeps subscriber callbacks alive without blocking the physics loop.
        self._spin_thread = threading.Thread(
            target=rospy.spin, daemon=True
        )
        self._spin_thread.start()

        rospy.loginfo("Shadow Hand digital twin initialised.")

    # ── Viewer thread ──────────────────────────────────────────────────────────

    def _run_viewer(self):
        with mujoco.viewer.launch_passive(self._model, self._data) as viewer:
            viewer.cam.lookat    = [0.2, 0, 0.2]
            viewer.cam.distance  = 0.8
            viewer.cam.elevation = -20
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            self._model.vis.map.force = 0.03
            # Signal that launch_passive has finished its internal mj_forward —
            # the physics loop may now safely apply the initial config.
            self._viewer_ready.set()

            while viewer.is_running() and not self._stop_event.is_set():
                viewer.sync()
                time.sleep(1 / 60)

        # Closing the viewer shuts down the node cleanly
        rospy.loginfo("Viewer closed — shutting down.")
        self._stop_event.set()
        rospy.signal_shutdown("Viewer closed.")

    # ── ROS subscriber callback ────────────────────────────────────────────────

    def _command_callback(self, msg: Float64MultiArray):
        """
        Receives 24 joint targets as a flat Float64MultiArray (in JOINT_NAMES
        order) and maps them to MuJoCo actuator ctrl indices.

        Coupling logic
        ──────────────
          Coupled model   : direct joints map 1-to-1; J1+J2 targets are summed
                            and written to the single J0 actuator.
          Uncoupled model : every value maps 1-to-1 to its rh_A_<joint> actuator;
                            no summation is performed.
        """
        if len(msg.data) != len(JOINT_NAMES):
            rospy.logwarn_throttle(
                5.0,
                f"Command message: expected {len(JOINT_NAMES)} values, "
                f"got {len(msg.data)}.",
            )
            return

        # Map flat array to joint names using the known JOINT_NAMES order
        cmd = {name: float(msg.data[i]) for i, name in enumerate(JOINT_NAMES)}

        ctrl = np.zeros(self._model.nu)

        if self._coupled:
            # 1. Direct actuators: joint name → rh_A_<joint> (1-to-1)
            for joint_name in JOINT_NAMES:
                actuator_name = joint_name.replace("rh_", "rh_A_")
                if actuator_name in self._actuator_ids:
                    ctrl[self._actuator_ids[actuator_name]] = cmd[joint_name]

            # 2. Coupled J0 actuators: sum J2 + J1 targets into the single J0 ctrl
            for actuator_name, (j2_name, j1_name) in COUPLED_JOINTS.items():
                ctrl[self._actuator_ids[actuator_name]] = cmd[j2_name] + cmd[j1_name]
        else:
            # Uncoupled: every joint maps 1-to-1 to its own actuator
            for joint_name in JOINT_NAMES:
                actuator_name = joint_name.replace("rh_", "rh_A_")
                if actuator_name in self._actuator_ids:
                    ctrl[self._actuator_ids[actuator_name]] = cmd[joint_name]

        with self._ctrl_lock:
            self._ctrl_buffer = ctrl

    # ── Lift callback ──────────────────────────────────────────────────────────

    def _lift_callback(self, msg: Empty):
        """
        Triggered by any message on /lift.
        Sets _lift_active=True; the physics loop then drives arm_A_lift
        to _lift_target. The flag stays True so the arm holds its height
        after reaching the target (the position actuator handles it).
        To lower the arm, publish a second message and extend this logic
        with a toggle or a separate /lower topic.
        """
        if not self._lift_active:
            rospy.loginfo("Lift triggered — moving arm to %.2f m.", self._lift_target)
            self._lift_active = True

    # ── Publisher helper ───────────────────────────────────────────────────────

    def _publish_state(self):
        """
        Reads the 24 hand joint positions and velocities and publishes them.

        The arm_lift prismatic joint is intentionally excluded — it is an
        infrastructure joint, not part of the Shadow Hand's joint state.
        """
        msg              = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name         = JOINT_NAMES
        msg.position     = [
            float(self._data.qpos[idx]) for idx in self._joint_qpos_ids
        ]
        msg.velocity     = [
            float(self._data.qvel[idx]) for idx in self._joint_qvel_ids
        ]
        self._state_pub.publish(msg)

    # ── Main physics loop ──────────────────────────────────────────────────────

    def run(self):
        """
        Physics loop running at model.opt.timestep rate (500 Hz).
        Joint states are published every PUBLISH_EVERY steps (125 Hz).

        Each iteration:
          1. Apply latest ctrl from ROS command buffer (thread-safe read)
          2. Step MuJoCo physics once
          3. Publish joint states every 4th step (500 Hz / 4 = 125 Hz)
          4. Sleep for the remainder of the timestep (precise_sleep)
        """
        dt            = self._model.opt.timestep   # 0.002 s  -> 500 Hz
        PHYSICS_HZ    = round(1.0 / dt)            # 500
        PUBLISH_HZ    = 125
        PUBLISH_EVERY = PHYSICS_HZ // PUBLISH_HZ   # 4  (publish every 4th step)

        rospy.loginfo(
            "Physics loop: %d Hz  |  state publish: %d Hz  (every %d steps).",
            PHYSICS_HZ, PUBLISH_HZ, PUBLISH_EVERY,
        )

        # Wait until the viewer has finished its internal mj_forward so our
        # apply_initial_config is the last thing written before the first step.
        rospy.loginfo("Waiting for viewer to initialise...")
        self._viewer_ready.wait()
        mujoco.mj_resetData(self._model, self._data)
        apply_initial_config(self._model, self._data, INITIAL_CONFIG, self._actuator_ids, self._coupled)
        rospy.loginfo("Viewer ready — starting physics loop.")

        last_time  = 0.0
        step_count = 0

        try:
            while not self._stop_event.is_set() and not rospy.is_shutdown():
                step_start = time.perf_counter()

                # Detect GUI reset (data.time jumps back to 0)
                if self._data.time < last_time:
                    rospy.loginfo("GUI reset detected - re-applying initial config.")
                    # mj_resetData wipes qpos, qvel, contacts and forces to a
                    # clean default state before we overwrite with our config.
                    # This prevents stale contact frames from causing
                    # mju_makeFrames fatal errors during the transition.
                    mujoco.mj_resetData(self._model, self._data)
                    apply_initial_config(
                        self._model, self._data, INITIAL_CONFIG, self._actuator_ids, self._coupled
                    )
                    step_count = 0
                    self._lift_active  = False
                    self._lift_current = INITIAL_CONFIG["rh_arm_lift"]  # 0.02
                    # Flush the command buffer so the hand holds the initial
                    # config rather than jumping back to the last ROS command.
                    with self._ctrl_lock:
                        self._ctrl_buffer = None
                    # Clear log so plots only show the latest run
                    self._logger.reset()
                last_time = self._data.time

                # 1. Apply ROS command -> data.ctrl
                with self._ctrl_lock:
                    ctrl_snapshot = self._ctrl_buffer

                if ctrl_snapshot is not None:
                    np.copyto(self._data.ctrl, ctrl_snapshot)

                # 2. Drive arm_A_lift independently of the hand command topic.
                #    Ramp _lift_current toward _lift_target at _lift_speed (m/s)
                #    so the motion is smooth rather than a step change in ctrl.
                lift_aid = self._actuator_ids["rh_A_arm_lift"]
                if self._lift_active and self._lift_current < self._lift_target:
                    step = self._lift_speed * dt   # metres advanced this physics step
                    self._lift_current = min(
                        self._lift_current + step, self._lift_target
                    )
                self._data.ctrl[lift_aid] = self._lift_current

                # 3. Step physics
                mujoco.mj_step(self._model, self._data)
                step_count += 1

                # 4. Log data for post-simulation plots
                self._logger.record()

                # 5. Publish joint states at 125 Hz (every 4th physics step)
                if step_count % PUBLISH_EVERY == 0:
                    self._publish_state()

                # 4. Real-time pacing
                elapsed   = time.perf_counter() - step_start
                remaining = dt - elapsed
                if remaining > 0:
                    precise_sleep(remaining)

        except KeyboardInterrupt:
            rospy.loginfo("Physics loop interrupted by user.")
        finally:
            self._stop_event.set()
            rospy.loginfo("Digital twin stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# 4. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Windows timer resolution
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
        _WINDOWS_TIMER = True
    except AttributeError:
        _WINDOWS_TIMER = False

    try:
        node = ShadowHandDigitalTwin()
        node.run()
    finally:
        if _WINDOWS_TIMER:
            ctypes.windll.winmm.timeEndPeriod(1)

    # Generate plots after the simulation ends (viewer closed or Ctrl-C)
    node._logger.plot()