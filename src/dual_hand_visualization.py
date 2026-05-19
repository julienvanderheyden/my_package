#!/usr/bin/env python3
"""
dual_hand_visualizer.py
=======================
ROS node that subscribes to:
  - /shadowhand_state_topic   → real hand joint positions (sensor feedback)
  - /shadowhand_command_topic → virtual hand joint targets (VMC planner output)

and renders both hands side-by-side in a PyBullet GUI at a fixed display rate
(default: 20 Hz) independent of topic publish rates.

Usage
-----
    rosrun <your_pkg> dual_hand_visualizer.py [--rate 20] [--urdf path/to/shadow_hand.urdf]

Requirements
------------
    pip install pybullet
    (ROS Noetic / Python 3 assumed; adapt imports for ROS2 if needed)

Joint name convention
---------------------
The node expects the joint names in the ShadowHand messages to match the
standard Shadow Dexterous Hand URDF naming, e.g.:
    rh_FFJ1, rh_FFJ2, rh_FFJ3, rh_FFJ4,
    rh_MFJ1 ... rh_LFJ5, rh_THJ1 ... rh_THJ5, rh_WRJ1, rh_WRJ2
Adjust JOINT_NAMES below if your URDF uses different conventions.
"""

from __future__ import annotations  # Enables Python 3.9+ type hinting syntax in Python 3.8

import argparse
import math
import os
import threading
import time

import pybullet as pb
import pybullet_data
import rospy
import rospkg

# ── Adapt these to your actual message types ──────────────────────────────────
# ShadowHand typically publishes sensor_msgs/JointState or a custom type.
# Change the import and the extractor functions below if yours differs.
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
# ─────────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Lateral offset between the two ghost hands in the visualizer (metres)
HAND_SEPARATION = 0.0

# Colours for the two hands (R, G, B, A)  – applied to all visual links
COLOR_REAL    = (0.20, 0.60, 1.00, 0.85)   # blue  – real hand (state topic)
COLOR_VIRTUAL = (1.00, 0.45, 0.10, 0.70)   # orange – virtual hand (command topic)

# Standard Shadow Dexterous Hand joint ordering (22 DoF, excluding coupled)
# Edit this list to match the joint_names in your URDF.
JOINT_NAMES = [
    "rh_WRJ1", "rh_WRJ2",
    "rh_FFJ1", "rh_FFJ2", "rh_FFJ3", "rh_FFJ4",
    "rh_MFJ1", "rh_MFJ2", "rh_MFJ3", "rh_MFJ4",
    "rh_RFJ1", "rh_RFJ2", "rh_RFJ3", "rh_RFJ4",
    "rh_LFJ1", "rh_LFJ2", "rh_LFJ3", "rh_LFJ4", "rh_LFJ5",
    "rh_THJ1", "rh_THJ2", "rh_THJ3", "rh_THJ4", "rh_THJ5",
]

# Default URDF path – override with --urdf argument
package_path = rospkg.RosPack().get_path('my_package')
DEFAULT_URDF = os.path.join(package_path, 'urdf/sr_hand_vm_compatible.urdf')

# ---------------------------------------------------------------------------
# State container (thread-safe via lock)
# ---------------------------------------------------------------------------

class HandState:
    """Holds the latest joint position array for one hand."""

    def __init__(self, n_joints: int):
        self._lock = threading.Lock()
        self._positions: dict[str, float] = {}
        self._stamp: float = 0.0
        self._received = False

    def update(self, name_position_pairs: list[tuple[str, float]]):
        with self._lock:
            for name, pos in name_position_pairs:
                self._positions[name] = pos
            self._stamp = time.monotonic()
            self._received = True

    def get(self) -> tuple[dict[str, float], float, bool]:
        with self._lock:
            return dict(self._positions), self._stamp, self._received


# ---------------------------------------------------------------------------
# Helper – extract joint name/position pairs from a JointState message
# ---------------------------------------------------------------------------

def extract_joint_state(msg: JointState) -> list[tuple[str, float]]:
    """Works for sensor_msgs/JointState.  Adapt for custom messages."""
    return list(zip(msg.name, msg.position))

def extract_joint_state_array(msg: Float64MultiArray) -> list[tuple[str, float]]:
    """Example extractor for a Float64MultiArray with joint names in the layout."""
    return list(zip(JOINT_NAMES, msg.data))


# ---------------------------------------------------------------------------
# PyBullet helpers
# ---------------------------------------------------------------------------

# def load_hand(urdf_path: str, base_position: list, color: tuple) -> tuple[int, dict[str, int]]:
#     """
#     Load the hand URDF at *base_position* and recolour all links.
#     Returns (body_id, joint_name_to_index_map).
#     """
#     body_id = pb.loadURDF(
#         urdf_path,
#         basePosition=base_position,
#         baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]),
#         useFixedBase=True,
#         flags=pb.URDF_USE_SELF_COLLISION | pb.URDF_MERGE_FIXED_LINKS,
#     )

#     joint_map: dict[str, int] = {}
#     n_joints = pb.getNumJoints(body_id)

#     for i in range(n_joints):
#         info = pb.getJointInfo(body_id, i)
#         name = info[1].decode("utf-8")
#         joint_type = info[2]

#         # Map only revolute / prismatic joints
#         if joint_type in (pb.JOINT_REVOLUTE, pb.JOINT_PRISMATIC):
#             joint_map[name] = i

#         # Recolour every link
#         pb.changeVisualShape(
#             body_id, i,
#             rgbaColor=color,
#         )

#     # Also recolour base link (-1)
#     pb.changeVisualShape(body_id, -1, rgbaColor=color)

#     return body_id, joint_map

def load_hand(urdf_path: str, base_position: list, color: tuple) -> tuple[int, dict[str, int]]:
    body_id = pb.loadURDF(
        urdf_path,
        basePosition=base_position,
        baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True,
        flags=pb.URDF_USE_SELF_COLLISION | pb.URDF_MERGE_FIXED_LINKS,
    )

    joint_map: dict[str, int] = {}
    n_joints = pb.getNumJoints(body_id)

    for i in range(n_joints):
        info = pb.getJointInfo(body_id, i)
        name = info[1].decode("utf-8")
        joint_type = info[2]

        if joint_type in (pb.JOINT_REVOLUTE, pb.JOINT_PRISMATIC):
            joint_map[name] = i
            
            # FIX: Disable the default motor control so physics solver doesn't fight you
            pb.setJointMotorControl2(
                bodyUniqueId=body_id,
                jointIndex=i,
                controlMode=pb.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0
            )

        pb.changeVisualShape(body_id, i, rgbaColor=color)

    pb.changeVisualShape(body_id, -1, rgbaColor=color)
    return body_id, joint_map


# def apply_positions(body_id: int,
#                     joint_map: dict[str, int],
#                     positions: dict[str, float]) -> None:
#     """Push joint positions into PyBullet (kinematic reset, no physics)."""
#     for name, idx in joint_map.items():
#         angle = positions.get(name, 0.0)
#         pb.resetJointState(body_id, idx, angle)

def apply_positions(body_id: int,
                    joint_map: dict[str, int],
                    positions: dict[str, float]) -> None:
    """Push joint positions into PyBullet (kinematic reset, no physics)."""
    for name, idx in joint_map.items():
        # FIX: Check if the joint name actually arrived in the ROS message
        if name in positions:
            angle = positions[name]
            pb.resetJointState(body_id, idx, angle)
        # If it's not in positions, LEAVE IT ALONE so mimic/default structures hold

# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

class DualHandVisualizer:

    def __init__(self, urdf_path: str, display_rate: float):
        self.urdf_path = urdf_path
        self.display_period = 1.0 / display_rate

        self.real_state    = HandState(len(JOINT_NAMES))
        self.virtual_state = HandState(len(JOINT_NAMES))

        # ROS subscribers
        rospy.Subscriber(
            "/shadowhand_state_topic",
            JointState,
            self._cb_real,
            queue_size=1,
        )
        # BUGFIX: Changed message type from JointState to Float64MultiArray to match the callback
        rospy.Subscriber(
            "/shadowhand_command_topic",
            Float64MultiArray,
            self._cb_virtual,
            queue_size=1,
        )

        rospy.loginfo("[DualHandVisualizer] Subscribed to state and command topics.")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cb_real(self, msg: JointState):
        self.real_state.update(extract_joint_state(msg))

    def _cb_virtual(self, msg: Float64MultiArray):
        self.virtual_state.update(extract_joint_state_array(msg))

    # ── PyBullet visualisation loop ───────────────────────────────────────────

    def run(self):
        # Start PyBullet with GUI
        client = pb.connect(pb.GUI)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(0, 0, 0)
        pb.resetDebugVisualizerCamera(
            cameraDistance=0.55,
            cameraYaw=30,
            cameraPitch=-20,
            cameraTargetPosition=[0, 0, 0.1],
        )

        # Nice dark background
        pb.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 1)

        # Load ground plane (optional, for spatial reference)
        pb.loadURDF("plane.urdf", [0, 0, -0.01], useFixedBase=True)

        # Load both hands side-by-side
        real_id,    real_jmap    = load_hand(
            self.urdf_path,
            base_position=[-HAND_SEPARATION / 2, 0, 0],
            color=COLOR_REAL,
        )
        virtual_id, virtual_jmap = load_hand(
            self.urdf_path,
            base_position=[+HAND_SEPARATION / 2, 0, 0],
            color=COLOR_VIRTUAL,
        )

        # On-screen labels (PyBullet debug text)
        txt_kwargs = dict(textSize=1.2, lifeTime=0)
        real_label = pb.addUserDebugText(
            "REAL  (state)",
            [-HAND_SEPARATION / 2, 0, 0.28],
            textColorRGB=[0.2, 0.6, 1.0],
            **txt_kwargs,
        )
        virt_label = pb.addUserDebugText(
            "VIRTUAL  (command)",
            [+HAND_SEPARATION / 2, 0, 0.28],
            textColorRGB=[1.0, 0.45, 0.1],
            **txt_kwargs,
        )

        # Status line (freshness indicator)
        status_id = pb.addUserDebugText(
            "Waiting for data …",
            [-0.05, 0, -0.05],
            textColorRGB=[0.8, 0.8, 0.8],
            textSize=0.9,
            lifeTime=0,
        )

        rospy.loginfo("[DualHandVisualizer] PyBullet GUI started. "
                      f"Blue = real hand  |  Orange = virtual hand")

        # ── Main display loop ──────────────────────────────────────────────────
        last_draw = time.monotonic()

        while not rospy.is_shutdown():
            now = time.monotonic()
            if now - last_draw < self.display_period:
                time.sleep(0.002)  # yield; do not busy-spin
                continue
            last_draw = now

            # Pull latest states
            real_pos,    real_t,    real_ok    = self.real_state.get()
            virtual_pos, virtual_t, virtual_ok = self.virtual_state.get()

            # Apply to PyBullet bodies
            if real_ok:
                apply_positions(real_id, real_jmap, real_pos)
            if virtual_ok:
                apply_positions(virtual_id, virtual_jmap, virtual_pos)

            # Update status text
            age_r = now - real_t    if real_ok    else float("inf")
            age_v = now - virtual_t if virtual_ok else float("inf")

            def age_str(a):
                if a == float("inf"):
                    return "no data"
                return f"{a*1000:.0f} ms ago"

            status_text = (
                f"Real: {age_str(age_r)}   |   Virtual: {age_str(age_v)}"
            )
            # Replace text (PyBullet has no update; remove + re-add)
            pb.removeUserDebugItem(status_id)
            status_id = pb.addUserDebugText(
                status_text,
                [-0.1, 0, -0.05],
                textColorRGB=[0.8, 0.8, 0.8],
                textSize=0.9,
                lifeTime=0,
            )

            # Step simulation (needed to refresh the GUI)
            pb.stepSimulation()

        pb.disconnect()
        rospy.loginfo("[DualHandVisualizer] Shutting down.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Dual ShadowHand visualizer")
    parser.add_argument(
        "--rate", type=float, default=20.0,
        help="Display refresh rate in Hz (default: 20)"
    )
    parser.add_argument(
        "--urdf", type=str, default=DEFAULT_URDF,
        help=f"Path to the Shadow Hand URDF (default: {DEFAULT_URDF})"
    )
    # Strip ROS remapping args before argparse sees them
    import sys
    args, _ = parser.parse_known_args(
        [a for a in sys.argv[1:] if not a.startswith("__")]
    )
    return args


if __name__ == "__main__":
    args = parse_args()
    rospy.init_node("dual_hand_visualizer", anonymous=False)
    node = DualHandVisualizer(urdf_path=args.urdf, display_rate=args.rate)
    node.run()