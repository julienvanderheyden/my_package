"""
shadow_hand_simulation.py
─────────────────────────
ROS-based digital twin for the Shadow Hand using MuJoCo.

ROS interface
─────────────
  PUBLISH  /shadowhand_state_topic   sensor_msgs/JointState   24 hand joints
  SUBSCRIBE /shadowhand_command_topic sensor_msgs/JointState   20 actuator targets

Joint ordering
──────────────
  Published joints  : JOINT_NAMES  (24 entries, read from qpos)
  Subscribed commands: ACTUATOR_NAMES (20 entries, written to data.ctrl)
  Coupled joints (J0): rh_A_FFJ0/MFJ0/RFJ0/LFJ0 each drive J2+J1 via tendon.
                        The subscriber sums J2+J1 and writes to the J0 actuator.
"""

import ctypes
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
import rospy
from sensor_msgs.msg import JointState

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

# 24 joints published on /shadowhand_state_topic (in qpos order)
JOINT_NAMES = [
    "rh_WRJ2", "rh_WRJ1",
    "rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_FFJ1",
    "rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1",
    "rh_RFJ4", "rh_RFJ3", "rh_RFJ2", "rh_RFJ1",
    "rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2", "rh_LFJ1",
    "rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1",
]

# 20 actuator commands received on /shadowhand_command_topic.
# The message's name[] field must match these strings exactly.
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
# The command message should provide the *summed* target (J2_target + J1_target).
COUPLED_ACTUATOR_NAMES = [
    "rh_A_FFJ0",   # drives rh_FFJ2 + rh_FFJ1
    "rh_A_MFJ0",   # drives rh_MFJ2 + rh_MFJ1
    "rh_A_RFJ0",   # drives rh_RFJ2 + rh_RFJ1
    "rh_A_LFJ0",   # drives rh_LFJ2 + rh_LFJ1
]

ALL_ACTUATOR_NAMES = DIRECT_ACTUATOR_NAMES + COUPLED_ACTUATOR_NAMES   # 20 total

# Coupled joints: used by apply_initial_config to compute J0 ctrl from J1+J2
COUPLED_JOINTS = {
    "rh_A_FFJ0": ("rh_FFJ2", "rh_FFJ1"),
    "rh_A_MFJ0": ("rh_MFJ2", "rh_MFJ1"),
    "rh_A_RFJ0": ("rh_RFJ2", "rh_RFJ1"),
    "rh_A_LFJ0": ("rh_LFJ2", "rh_LFJ1"),
}

# Initial hand pose (joint_name → angle in rad)
INITIAL_CONFIG = {
    "rh_FFJ4": 0.0, "rh_FFJ3": 0.0, "rh_FFJ2": 0.0, "rh_FFJ1": 0.0,
    "rh_MFJ4": 0.0, "rh_MFJ3": 0.0, "rh_MFJ2": 0.0, "rh_MFJ1": 0.0,
    "rh_RFJ4": 0.0, "rh_RFJ3": 0.0, "rh_RFJ2": 0.0, "rh_RFJ1": 0.0,
    "rh_LFJ5": 0.0, "rh_LFJ4": 0.0, "rh_LFJ3": 0.0, "rh_LFJ2": 0.0, "rh_LFJ1": 0.0,
    "rh_THJ5": 0.0, "rh_THJ4": 1.21, "rh_THJ3": 0.0, "rh_THJ2": 0.0, "rh_THJ1": 0.0,
    "rh_WRJ2": 0.0, "rh_WRJ1": 0.0,
    "rh_arm_lift": 0.02,
}


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_index_maps(model):
    """
    Pre-compute integer indices so the hot loop never calls mj_name2id.

    Returns
    -------
    joint_qpos_ids   : list[int]  qpos address for each entry in JOINT_NAMES
    actuator_ids     : dict[str, int]  actuator name → ctrl index
    """
    joint_qpos_ids = []
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid == -1:
            raise RuntimeError(f"Joint '{name}' not found in model.")
        joint_qpos_ids.append(model.jnt_qposadr[jid])

    actuator_ids = {}
    for name in ALL_ACTUATOR_NAMES:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid == -1:
            raise RuntimeError(f"Actuator '{name}' not found in model.")
        actuator_ids[name] = aid

    return joint_qpos_ids, actuator_ids


def apply_initial_config(model, data, config: dict, actuator_ids: dict):
    """Set qpos and ctrl so the hand starts in `config` without fighting itself."""

    # 1. All joint positions
    for joint_name, angle in config.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid != -1:
            data.qpos[model.jnt_qposadr[jid]] = angle

    # 2. Direct actuators (joint_name → rh_A_<joint_name>)
    for joint_name, angle in config.items():
        actuator_name = joint_name.replace("rh_", "rh_A_")
        if actuator_name in actuator_ids:
            data.ctrl[actuator_ids[actuator_name]] = angle

    # 3. Coupled J0 actuators: ctrl = J2_angle + J1_angle
    for actuator_name, (j2_name, j1_name) in COUPLED_JOINTS.items():
        if actuator_name in actuator_ids:
            j2 = config.get(j2_name, 0.0)
            j1 = config.get(j1_name, 0.0)
            data.ctrl[actuator_ids[actuator_name]] = j2 + j1

    mujoco.mj_forward(model, data)


# ══════════════════════════════════════════════════════════════════════════════
# 3. ROS NODE
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
            JointState,
            self._command_callback,
            queue_size=1,
        )

        # ── MuJoCo model ──────────────────────────────────────────────────────
        self._model = mujoco.MjModel.from_xml_path("shadow_hand/scene_right_perso.xml")
        self._data  = mujoco.MjData(self._model)

        # Pre-compute index maps (no name lookups in the hot loop)
        self._joint_qpos_ids, self._actuator_ids = build_index_maps(self._model)

        apply_initial_config(
            self._model, self._data, INITIAL_CONFIG, self._actuator_ids
        )

        # ── Viewer ────────────────────────────────────────────────────────────
        self._stop_event  = threading.Event()
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

            while viewer.is_running() and not self._stop_event.is_set():
                viewer.sync()
                time.sleep(1 / 60)

        # Closing the viewer shuts down the node cleanly
        rospy.loginfo("Viewer closed — shutting down.")
        self._stop_event.set()
        rospy.signal_shutdown("Viewer closed.")

    # ── ROS subscriber callback ────────────────────────────────────────────────

    def _command_callback(self, msg: JointState):
        """
        Receives 20 actuator targets and stores them in a thread-safe buffer.

        Message format
        ──────────────
          msg.name[]     : actuator names (must match ALL_ACTUATOR_NAMES)
          msg.position[] : target position for each actuator (rad / m)

        The physics loop reads this buffer once per step and writes data.ctrl.
        Using a lock + copy avoids partial writes from concurrent access.
        """
        if len(msg.name) != len(msg.position):
            rospy.logwarn_throttle(5.0, "Command message: name/position length mismatch.")
            return

        # Build a name→value dict from the incoming message
        cmd = {name: pos for name, pos in zip(msg.name, msg.position)}

        # Validate all expected actuators are present
        missing = [n for n in ALL_ACTUATOR_NAMES if n not in cmd]
        if missing:
            rospy.logwarn_throttle(5.0, f"Command missing actuators: {missing}")
            return

        # Write ordered ctrl array (indexed by actuator order in self._actuator_ids)
        ctrl = np.zeros(self._model.nu)
        for name, aid in self._actuator_ids.items():
            ctrl[aid] = cmd[name]

        with self._ctrl_lock:
            self._ctrl_buffer = ctrl

    # ── Publisher helper ───────────────────────────────────────────────────────

    def _publish_state(self):
        """
        Reads the 24 hand joint positions from qpos and publishes them.

        The arm_lift prismatic joint is intentionally excluded — it is an
        infrastructure joint, not part of the Shadow Hand's joint state.
        """
        msg              = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name         = JOINT_NAMES
        msg.position     = [
            float(self._data.qpos[idx]) for idx in self._joint_qpos_ids
        ]
        # Velocities and efforts are also available if needed:
        # msg.velocity = [float(self._data.qvel[...]) for ...]
        # msg.effort   = [float(self._data.qfrc_actuator[...]) for ...]
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

        last_time  = 0.0
        step_count = 0

        try:
            while not self._stop_event.is_set() and not rospy.is_shutdown():
                step_start = time.perf_counter()

                # Detect GUI reset (data.time jumps back to 0)
                if self._data.time < last_time:
                    rospy.loginfo("GUI reset detected - re-applying initial config.")
                    apply_initial_config(
                        self._model, self._data, INITIAL_CONFIG, self._actuator_ids
                    )
                    step_count = 0
                last_time = self._data.time

                # 1. Apply ROS command -> data.ctrl
                with self._ctrl_lock:
                    ctrl_snapshot = self._ctrl_buffer

                if ctrl_snapshot is not None:
                    np.copyto(self._data.ctrl, ctrl_snapshot)

                # 2. Step physics
                mujoco.mj_step(self._model, self._data)
                step_count += 1

                # 3. Publish joint states at 125 Hz (every 4th physics step)
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