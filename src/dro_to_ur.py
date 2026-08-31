#!/usr/bin/env python3
"""
dro_to_ur.py
============
Bridges D(R,O) Grasp inference output to the real ShadowHand + arm.
"""

import sys
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
# Constant: rh_forearm → rh_manipulator  (from URDF / live tf)
# ═══════════════════════════════════════════════════════════════════════════════

T_FOREARM_TO_MANIPULATOR_XYZ  = np.array([0.001, -0.002, 0.296])
T_FOREARM_TO_MANIPULATOR_QUAT = np.array([-0.077, 0.003, 0.0, 0.997])  # [x,y,z,w]


# ═══════════════════════════════════════════════════════════════════════════════
# Manual Perturbation / Offset Parameters (Editable)
# ═══════════════════════════════════════════════════════════════════════════════

# Specify exact translation offsets [dx, dy, dz] in meters
OFFSET_XYZ = np.array([0.01, -0.005, 0.000])  # e.g. +1cm in X, -5mm in Y

# Specify exact rotation offsets [roll, pitch, yaw] in radians (or use np.radians(deg))
OFFSET_RPY = np.array([np.radians(0.0), np.radians(0.0), np.radians(5.0)])  # e.g. +5 degrees yaw offset


# ═══════════════════════════════════════════════════════════════════════════════
# Finger joint names
# ═══════════════════════════════════════════════════════════════════════════════

DRO_JOINT_ORDER = [
    "rh_WRJ2", "rh_WRJ1",
    "rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_FFJ1",
    "rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1",
    "rh_RFJ4", "rh_RFJ3", "rh_RFJ2", "rh_RFJ1",
    "rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2", "rh_LFJ1",
    "rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1",
]

CMD_JOINT_ORDER = [
    "rh_WRJ1", "rh_WRJ2",
    "rh_FFJ1", "rh_FFJ2", "rh_FFJ3", "rh_FFJ4",
    "rh_MFJ1", "rh_MFJ2", "rh_MFJ3", "rh_MFJ4",
    "rh_RFJ1", "rh_RFJ2", "rh_RFJ3", "rh_RFJ4",
    "rh_LFJ1", "rh_LFJ2", "rh_LFJ3", "rh_LFJ4", "rh_LFJ5",
    "rh_THJ1", "rh_THJ2", "rh_THJ3", "rh_THJ4", "rh_THJ5",
]

assert len(DRO_JOINT_ORDER) == 24
assert len(CMD_JOINT_ORDER) == 24
assert set(DRO_JOINT_ORDER) == set(CMD_JOINT_ORDER), "Joint name mismatch!"

_dro_pos = {name: i for i, name in enumerate(DRO_JOINT_ORDER)}
REINDEX = [_dro_pos[name] for name in CMD_JOINT_ORDER]


# ═══════════════════════════════════════════════════════════════════════════════
# Math helpers
# ═══════════════════════════════════════════════════════════════════════════════

def Rx(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1,  0,   0,  0],
                     [0,  ca, -sa, 0],
                     [0,  sa,  ca, 0],
                     [0,  0,   0,  1]], dtype=float)

def Ry(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ ca, 0, sa, 0],
                     [  0, 1,  0, 0],
                     [-sa, 0, ca, 0],
                     [  0, 0,  0, 1]], dtype=float)

def Rz(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0, 0],
                     [sa,  ca, 0, 0],
                     [ 0,   0, 1, 0],
                     [ 0,   0, 0, 1]], dtype=float)

def xyz_quat_to_matrix(xyz, quat):
    T = tft.quaternion_matrix(quat)
    T[0:3, 3] = xyz
    return T

def xyz_rpy_to_matrix(xyz, rpy):
    T = tft.euler_matrix(rpy[0], rpy[1], rpy[2], axes='sxyz')
    T[0:3, 3] = xyz
    return T

def matrix_to_xyz_quat(T):
    return T[0:3, 3].copy(), tft.quaternion_from_matrix(T)


# ═══════════════════════════════════════════════════════════════════════════════
# Core: reconstruct T_object_forearm from q[0:6]
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruct_T_object_forearm(q):
    T_trans = np.eye(4)
    T_trans[0, 3] = q[0]
    T_trans[1, 3] = q[1]
    T_trans[2, 3] = q[2]

    T_rot = Rx(q[3]) @ Ry(q[4]) @ Rz(q[5])
    return T_trans @ T_rot


def dro_q_to_world_manipulator(q, T_world_object, T_forearm_manipulator):
    T_object_forearm    = reconstruct_T_object_forearm(q)
    T_world_forearm     = T_world_object @ T_object_forearm
    T_world_manipulator = T_world_forearm @ T_forearm_manipulator

    xyz, quat = matrix_to_xyz_quat(T_world_manipulator)
    return xyz, quat, q[6:30]


def apply_offset(xyz, quat, delta_xyz, delta_rpy):
    """Applies exact deterministic positional and RPY rotational offsets."""
    # Add translation offset
    offset_xyz = xyz + delta_xyz

    # Apply rotational offset in local/world frame
    delta_R = tft.euler_matrix(delta_rpy[0], delta_rpy[1], delta_rpy[2], axes='sxyz')
    T_orig = tft.quaternion_matrix(quat)
    T_offset = T_orig @ delta_R
    offset_quat = tft.quaternion_from_matrix(T_offset)

    return offset_xyz, offset_quat


# ═══════════════════════════════════════════════════════════════════════════════
# ROS executor
# ═══════════════════════════════════════════════════════════════════════════════

class DROArmExecutor:

    def __init__(self, predicted_grasp, grasp_outer, grasp_inner, T_world_object):
        self.grasp       = predicted_grasp   # (30,)
        self.grasp_outer = grasp_outer       # (30,)
        self.grasp_inner = grasp_inner       # (30,)
        self.T_world_obj = T_world_object    # (4,4)

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
            "/shadowhand_command_filtering",
            Float64MultiArray, queue_size=10,
        )

        rospy.Service("/dro/execute_grasp", Trigger, self._cb_execute_grasp)
        rospy.loginfo("DROArmExecutor ready.")

    def publish_hand_joints(self, joints_dro_order):
        """Reorders and publishes joint angles to the hand command topic."""
        joints_cmd = [float(joints_dro_order[i]) for i in REINDEX]
        msg = Float64MultiArray()
        msg.data = joints_cmd
        self._joint_pub.publish(msg)

    def move_arm_cartesian(self, xyz, quat):
        """Helper to send a Cartesian move command to the arm service."""
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
            if not resp.success:
                rospy.logwarn(f"Cartesian path {resp.fraction_complete*100:.1f}% complete. {resp.message}")
                return False
            return True
        except rospy.ServiceException as e:
            rospy.logerr(f"MoveCartesian call failed: {e}")
            return False

    def execute_grasp(self, q):
        # ---------------------------------------------------------------------
        # STEP 1: Place all hand joints to zero for preshaping
        # ---------------------------------------------------------------------
        rospy.loginfo("Step 1: Setting hand joints to zero for preshaping...")
        zero_joints = np.zeros(24)
        self.publish_hand_joints(zero_joints)
        rospy.sleep(1.0)  # Short pause to let preshape complete

        # ---------------------------------------------------------------------
        # STEP 2: Reach the correct pose for the arm
        # ---------------------------------------------------------------------
        xyz, quat, _ = dro_q_to_world_manipulator(
            q, self.T_world_obj, self.T_forearm_manipulator
        )

        rospy.loginfo(f"Step 2: Reaching target arm pose...\n  xyz: {np.round(xyz, 4)}\n  quat: {np.round(quat, 4)}")
        if not self.move_arm_cartesian(xyz, quat):
            rospy.logerr("Failed to reach target arm pose.")
            return False

        # ---------------------------------------------------------------------
        # STEP 3: Wait 10 seconds for user placement
        # ---------------------------------------------------------------------
        rospy.loginfo("Step 3: Waiting 10 seconds for user to place the object...")
        rospy.sleep(10.0)

        # ---------------------------------------------------------------------
        # STEP 4: Apply controlled deterministic perturbation on the arm position
        # ---------------------------------------------------------------------
        rospy.loginfo(f"Step 4: Applying configured offset:\n  XYZ Delta: {OFFSET_XYZ}\n  RPY Delta (rad): {OFFSET_RPY}")
        offset_xyz, offset_quat = apply_offset(xyz, quat, OFFSET_XYZ, OFFSET_RPY)
        rospy.loginfo(f"  Target Offset xyz: {np.round(offset_xyz, 4)}\n  Target Offset quat: {np.round(offset_quat, 4)}")

        if not self.move_arm_cartesian(offset_xyz, offset_quat):
            rospy.logerr("Failed to apply offset arm motion.")
            return False

        # ---------------------------------------------------------------------
        # STEP 5: Run three grasp sequences with 1.5s interval
        # ---------------------------------------------------------------------
        rospy.loginfo("Step 5: Executing sequential grasps (Outer -> Mid -> Inner)...")
        
        _, _, joints_outer = dro_q_to_world_manipulator(self.grasp_outer, self.T_world_obj, self.T_forearm_manipulator)
        _, _, joints_mid   = dro_q_to_world_manipulator(self.grasp,       self.T_world_obj, self.T_forearm_manipulator)
        _, _, joints_inner = dro_q_to_world_manipulator(self.grasp_inner, self.T_world_obj, self.T_forearm_manipulator)

        # Grasp 1: Outer
        rospy.loginfo("  Executing outer grasp...")
        self.publish_hand_joints(joints_outer)
        rospy.sleep(1.5)

        # Grasp 2: Mid
        rospy.loginfo("  Executing mid grasp...")
        self.publish_hand_joints(joints_mid)
        rospy.sleep(1.5)

        # Grasp 3: Inner
        rospy.loginfo("  Executing inner grasp...")
        self.publish_hand_joints(joints_inner)
        rospy.sleep(1.5)

        # ---------------------------------------------------------------------
        # STEP 6: Lift the object by sending reference +20cm on Z axis
        # ---------------------------------------------------------------------
        rospy.loginfo("Step 6: Lifting object +20cm along Z-axis...")
        try:
            current_pose_resp = self._get_pose(GetPoseRequest())
            current_pose = current_pose_resp.pose
            
            lift_xyz = np.array([
                current_pose.position.x,
                current_pose.position.y,
                current_pose.position.z + 0.20  # +20 cm along Z axis
            ])
            
            lift_quat = np.array([
                current_pose.orientation.x,
                current_pose.orientation.y,
                current_pose.orientation.z,
                current_pose.orientation.w
            ])

            if not self.move_arm_cartesian(lift_xyz, lift_quat):
                rospy.logerr("Failed to execute lift motion.")
                return False

        except rospy.ServiceException as e:
            rospy.logerr(f"GetPose call failed: {e}")
            return False

        rospy.loginfo("Grasp execution and lift completed successfully!")
        return True

    def _cb_execute_grasp(self, req):
        success = self.execute_grasp(self.grasp)
        return TriggerResponse(
            success=success,
            message="ok" if success else "FAILED",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostic: print the reconstructed forearm pose for a given q
# ═══════════════════════════════════════════════════════════════════════════════

def print_reconstruction_diagnostic(q):
    T = reconstruct_T_object_forearm(q)
    xyz, quat = matrix_to_xyz_quat(T)
    rpy = tft.euler_from_matrix(T, axes='sxyz')
    print("\n── Reconstruction diagnostic ─────────────────────────────────")
    print(f"  q[0:3] (translation)     : {q[0:3]}")
    print(f"  q[3:6] (joint angles)    : {q[3:6]}  rad")
    print(f"  Reconstructed xyz        : {np.round(xyz, 4)}")
    print(f"  Reconstructed quat(xyzw) : {np.round(quat, 4)}")
    print(f"  Reconstructed RPY (check): {np.round(rpy, 4)}  rad")
    print(f"  Rotation matrix R :\n{np.round(T[:3,:3], 4)}")
    print("──────────────────────────────────────────────────────────────\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():

    test_cylinder_5cm = np.array([[-0.2464, -0.1458, 0.2364, -1.9569, 0.8954, -3.1595, -0.3837, 
    0.3708, -0.2477, 0.5914, 0.1563, -0.0, -0.1317, 0.3416, 1.0397, 0.0, 0.0339, 0.8938, 0.0, 
    1.1701, 0.3513, -0.0555, 0.7188, 0.6032, 0.2067, -0.4843, 0.7217, 0.1398, 0.1412, 0.8045], 
    [-0.2464, -0.1458, 0.2364, -1.9569, 0.8954, -3.1595, -0.3837, 0.3708, -0.0985, 0.3781, 0.1173, 
     -0.0, -0.0115, 0.1907, 0.7798, 0.0, -0.0618, 0.6049, 0.0, 1.2702, 0.2635, -0.1289, 0.4736, 0.4524, 
     0.5477, -0.625, 0.5412, 0.1572, -0.0686, 0.5379], [-0.2464, -0.1458, 0.2364, -1.9569, 0.8954, -3.1595, 
    -0.3837, 0.3708, -0.2629, 0.7383, 0.3685, 0.2356, -0.1643, 0.526, 1.1194, 0.2356, 0.0812, 0.9954, 0.2356, 
    0.9945, 0.4164, 0.0052, 0.8466, 0.7483, 0.1757, -0.2546, 0.7967, 0.0874, 0.2248, 0.9195]])
    
    grasps = test_cylinder_5cm

    grasp = grasps[0]
    grasp_outer = grasps[1]
    grasp_inner = grasps[2]

    print_reconstruction_diagnostic(grasp)

    object_xyz = [1.279, 0.14, 0.74+0.085]
    object_rpy = [0.0, 0.0, 0.0]

    T_world_object = xyz_rpy_to_matrix(
        np.array(object_xyz, dtype=float),
        np.array(object_rpy, dtype=float),
    )

    executor = DROArmExecutor(
        predicted_grasp=grasp,
        grasp_outer=grasp_outer,
        grasp_inner=grasp_inner,
        T_world_object=T_world_object,
    )

    rospy.spin()


if __name__ == "__main__":
    main()