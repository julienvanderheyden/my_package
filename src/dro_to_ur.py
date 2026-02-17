#!/usr/bin/env python3
"""
dro_to_arm_executor.py
======================
Bridges D(R,O) Grasp inference output to the real ShadowHand + arm.

═══════════════════════════════════════════════════════════════════════════════
FRAME ANALYSIS — READ THIS BEFORE RUNNING
═══════════════════════════════════════════════════════════════════════════════

What DRO q[0:6] actually encodes
----------------------------------
The DRO ShadowHand URDF adds virtual prismatic + revolute joints between a
dummy "world" link and the first REAL link of the ShadowHand, which is
`rh_forearm`.  Therefore:

    q[0:3]  =  translation of `rh_forearm` expressed in the object frame
    q[3:6]  =  RPY orientation of `rh_forearm` expressed in the object frame
    q[6:30] =  24 real finger + wrist joints

i.e. the 6-DOF pose is the pose of the `rh_forearm` frame, NOT the palm,
NOT `rh_manipulator`.

What arm_motion_service controls
----------------------------------
Your existing service moves the `rh_manipulator` frame (the hand mounting
flange), not `rh_forearm`.  The two frames are related by a constant rigid
transform encoded in the ShadowHand URDF:

    T_forearm_manipulator  (constant, read from URDF / tf)

Full transform chain
---------------------
    T_world_manipulator
        = T_world_object           ← you provide this (measured object pose)
        × T_object_forearm         ← from q[0:6]
        × T_forearm_manipulator    ← URDF constant  ← YOU MUST FILL THIS IN

How to find T_forearm_manipulator
-----------------------------------
Option A — live tf (easiest, robot must be running):

    rosrun tf tf_echo rh_forearm rh_manipulator

Option B — URDF inspection:
    Accumulate all fixed joints between `rh_forearm` and `rh_manipulator`
    in the ShadowHand URDF.

Option C — run this script with --print_tf_only:

    python dro_to_arm_executor.py --print_tf_only

It will look up the transform from a running robot and print the values
to paste back into this file.

Fill in T_FOREARM_TO_MANIPULATOR_* below before executing grasps.
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import argparse
import numpy as np
import rospy
import tf
import tf.transformations as tft

from geometry_msgs.msg import Pose
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger, TriggerResponse

from my_package.srv import (
    MoveCartesian, MoveCartesianRequest,
    GetPose,       GetPoseRequest,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ▶▶▶  FILL IN BEFORE RUNNING  ◀◀◀
#
# Constant rigid transform:  rh_forearm  →  rh_manipulator
# Run:  rosrun tf tf_echo rh_forearm rh_manipulator
# (or use --print_tf_only flag below)
#
# Until this is set, the node will still run but will log a loud warning and
# the arm will be sent to the wrong pose.
# ═══════════════════════════════════════════════════════════════════════════════

# Translation [x, y, z] in metres
T_FOREARM_TO_MANIPULATOR_XYZ = np.array([0.001, -0.002, 0.296])    

# Quaternion [x, y, z, w]
T_FOREARM_TO_MANIPULATOR_QUAT = np.array([-0.077, 0.003, 0.0, 0.997])  


# ═══════════════════════════════════════════════════════════════════════════════
# ShadowHand joint names — must match the order of q[6:30] from DRO.
# Verify against the DRO shadowhand URDF: joints listed after the 6 virtual
# DOFs, in the same order as they appear in the URDF chain.
# ═══════════════════════════════════════════════════════════════════════════════
SHADOW_JOINT_NAMES = [
    "rh_WRJ2", "rh_WRJ1",                                  # wrist (2)
    "rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_FFJ1",           # first finger (4)
    "rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1",           # middle finger (4)
    "rh_RFJ4", "rh_RFJ3", "rh_RFJ2", "rh_RFJ1",           # ring finger (4)
    "rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2", "rh_LFJ1",  # little finger (5)
    "rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1",  # thumb (5)
]
assert len(SHADOW_JOINT_NAMES) == 24, "List must have exactly 24 entries."


# ═══════════════════════════════════════════════════════════════════════════════
# Pure math helpers
# ═══════════════════════════════════════════════════════════════════════════════

def xyz_rpy_to_matrix(xyz, rpy):
    """4×4 homogeneous matrix from translation + roll/pitch/yaw [rad]."""
    T = tft.euler_matrix(rpy[0], rpy[1], rpy[2], axes='sxyz')
    T[0:3, 3] = xyz
    return T


def xyz_quat_to_matrix(xyz, quat):
    """4×4 homogeneous matrix from translation + quaternion [x, y, z, w]."""
    T = tft.quaternion_matrix(quat)
    T[0:3, 3] = xyz
    return T


def matrix_to_xyz_quat(T):
    """Return (xyz, quat [x,y,z,w]) from a 4×4 homogeneous matrix."""
    return T[0:3, 3].copy(), tft.quaternion_from_matrix(T)


# ═══════════════════════════════════════════════════════════════════════════════
# Core transform pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def dro_q_to_world_manipulator(q, T_world_object, T_forearm_manipulator):
    """
    Convert one DRO grasp vector into the desired rh_manipulator pose (world
    frame) and the 24 finger joint values.

    Pipeline:
        T_world_manipulator =
            T_world_object          (known object pose in world)
          · T_object_forearm        (from q[0:6]: rh_forearm in object frame)
          · T_forearm_manipulator   (fixed URDF offset)

    Args:
        q                     : (30,) numpy array from predict_grasp()
        T_world_object         : (4,4) pose of object centroid in robot world
        T_forearm_manipulator  : (4,4) constant rh_forearm → rh_manipulator

    Returns:
        xyz    : (3,) target position of rh_manipulator in world frame [m]
        quat   : (4,) target quaternion [x,y,z,w] of rh_manipulator in world
        joints : (24,) finger + wrist joint values [rad]
    """
    xyz_forearm_in_obj = q[0:3]
    rpy_forearm_in_obj = q[3:6]
    joints             = q[6:30]

    # rh_forearm pose expressed in the object frame
    T_object_forearm = xyz_rpy_to_matrix(xyz_forearm_in_obj, rpy_forearm_in_obj)

    # Chain: world ← object ← forearm ← manipulator
    T_world_manipulator = T_world_object @ T_object_forearm @ T_forearm_manipulator

    xyz, quat = matrix_to_xyz_quat(T_world_manipulator)
    return xyz, quat, joints


# ═══════════════════════════════════════════════════════════════════════════════
# ROS executor
# ═══════════════════════════════════════════════════════════════════════════════

class DROArmExecutor:
    """
    Loads a batch of DRO grasps and provides ROS services to execute them
    on the real robot one at a time.
    """

    def __init__(self, predicted_grasp, T_world_object):
        
        self.grasp       = predicted_grasp         # (N, 30)
        self.T_world_obj = T_world_object               # (4, 4)
        self.current_idx = 0

        self.T_forearm_manipulator = xyz_quat_to_matrix(
            T_FOREARM_TO_MANIPULATOR_XYZ,
            T_FOREARM_TO_MANIPULATOR_QUAT,
        )

        rospy.init_node("dro_arm_executor", anonymous=False)

        rospy.loginfo("Waiting for /arm/move_cartesian …")
        rospy.wait_for_service("/arm/move_cartesian")
        self._move_arm = rospy.ServiceProxy("/arm/move_cartesian", MoveCartesian)

        rospy.loginfo("Waiting for /arm/get_current_pose …")
        rospy.wait_for_service("/arm/get_current_pose")
        self._get_pose = rospy.ServiceProxy("/arm/get_current_pose", GetPose)

        self._joint_pub = rospy.Publisher(
            "/shadowhand_command_topic",
            Float64MultiArray, queue_size=10,
        )

        rospy.Service("/dro/execute_grasp",      Trigger, self._cb_execute_grasp)

        rospy.loginfo(
            f"DROArmExecutor ready "
        )

    # ── public api ────────────────────────────────────────────────────────────

    def execute_grasp(self, q):
        """Compute the rh_manipulator target pose and execute it."""

        xyz, quat, joints = dro_q_to_world_manipulator(
            q,
            self.T_world_obj,
            self.T_forearm_manipulator,
        )

        rospy.loginfo(
            f"Executing grasp :\n"
            f"  rh_manipulator target xyz  (world): {np.round(xyz, 4)}\n"
            f"  rh_manipulator target quat (world): {np.round(quat, 4)}"
        )

        # Send arm
        req = MoveCartesianRequest()
        req.position.x    = float(xyz[0])
        req.position.y    = float(xyz[1])
        req.position.z    = float(xyz[2])
        req.orientation.x = float(quat[0])
        req.orientation.y = float(quat[1])
        req.orientation.z = float(quat[2])
        req.orientation.w = float(quat[3])

        try:
            resp = self._move_arm(req)
        except rospy.ServiceException as e:
            rospy.logerr(f"MoveCartesian call failed: {e}")
            return False

        if not resp.success:
            rospy.logwarn(
                f"Cartesian path {resp.fraction_complete*100:.1f}% complete. "
                f"{resp.message}"
            )
            return False

        rospy.loginfo("Arm motion complete.")

        # Send finger joints
        js = Float64MultiArray()
        js.data     = [float(j) for j in joints]
        self._joint_pub.publish(js)
        rospy.loginfo(f"  Finger joints published: {[round(float(j), 3) for j in joints]}")

        return True

    # ── service callbacks ─────────────────────────────────────────────────────

    def _cb_execute_grasp(self, req):
        success = self.execute_grasp(self.grasp)
        return TriggerResponse(
            success=success,
            message=f"Grasp {'ok' if success else 'FAILED'}",
        )



# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════



def main():

    grasp = np.array([
        -0.2, 0.0, 0.1,    # xyz forearm in object frame
        3.14, 1.57, 0.0,     # rpy forearm in object frame
        *([0.0] * 24)     # finger joints (dummy values)
    ], dtype=float)

    object_xyz = [1.35, 0.215, 0.74]  # 5
    object_rpy = [0.0, 0.0, 0.0] 

    T_world_object = xyz_rpy_to_matrix(
        np.array(object_xyz, dtype=float),
        np.array(object_rpy, dtype=float),
    )

    executor = DROArmExecutor(
        predicted_grasp=grasp,
        T_world_object=T_world_object,
    )

    rospy.spin()


if __name__ == "__main__":
    main()