#!/usr/bin/env python3
"""
dro_to_ur.py
============
Bridges D(R,O) Grasp inference output to the real ShadowHand + arm using
the reaching_service interface.
"""

import sys
import os
import numpy as np
import rospy
import rospkg
import tf
import tf.transformations as tft
import tf2_ros

from geometry_msgs.msg import Pose
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger, TriggerResponse

from my_package.srv import MoveToPose, MoveToPoseRequest

# ═══════════════════════════════════════════════════════════════════════════════
# Constant Transforms
# ═══════════════════════════════════════════════════════════════════════════════

# 1. rh_forearm → rh_manipulator (from URDF / live tf)
#    This is a rigid mounting offset (arm flange <-> hand base), independent of joints.
T_FOREARM_TO_MANIPULATOR_XYZ  = np.array([0.001, -0.002, 0.296])
T_FOREARM_TO_MANIPULATOR_QUAT = np.array([-0.077, 0.003, 0.0, 0.997])  # [x,y,z,w]

# 2. ra_flange → rh_manipulator (from arm_motion_service)
T_FLANGE_TO_MANIPULATOR_XYZ = [0.297, 0.000, 0.010]
T_FLANGE_TO_MANIPULATOR_RPY = [-1.575, 0.000, -1.563]  # Euler angles rads

# NOTE: rh_forearm -> rh_palm is NOT a constant transform: it is crossed by
# rh_WRJ2 then rh_WRJ1, which are not necessarily zeroed. Rather than
# forward-kinematting it from commanded joint angles, it is looked up LIVE
# from tf at the moment it's needed -- this reflects the hand's actual
# measured wrist configuration (tendon friction/backlash included), not the
# idealized commanded one.
HAND_BASE_FRAME = "rh_forearm"
EE_FRAME        = "rh_palm"


# ═══════════════════════════════════════════════════════════════════════════════
# Manual Perturbation / Offset Parameters (Editable)
# ═══════════════════════════════════════════════════════════════════════════════

# Specify exact translation offsets [dx, dy, dz] in meters
OFFSET_XYZ = np.array([0.0 , 0.0, 0.0])  

# Specify exact rotation offsets [around x, around y, around z] in radians (or use np.radians(deg))
OFFSET_RPY = np.array([np.radians(0.0), np.radians(0.0), np.radians(15.0)]) 


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


def apply_local_offset(xyz, quat, delta_xyz, delta_rpy):
    """Applies a translation+rotation offset expressed in the LOCAL frame of (xyz, quat).

    T_new = T_orig @ T_delta, so delta_xyz/delta_rpy are interpreted along the
    axes of the frame being perturbed (e.g. rh_palm), not world axes.
    """
    T_orig  = xyz_quat_to_matrix(xyz, quat)
    T_delta = xyz_rpy_to_matrix(np.asarray(delta_xyz, dtype=float),
                                 np.asarray(delta_rpy, dtype=float))
    T_new = T_orig @ T_delta
    return matrix_to_xyz_quat(T_new)


# ═══════════════════════════════════════════════════════════════════════════════
# Core: reconstruct T_object_forearm and T_world_flange
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruct_T_object_forearm(q):
    T_trans = np.eye(4)
    T_trans[0, 3] = q[0]
    T_trans[1, 3] = q[1]
    T_trans[2, 3] = q[2]

    T_rot = Rx(q[3]) @ Ry(q[4]) @ Rz(q[5])
    return T_trans @ T_rot


def dro_q_to_world_flange(q, T_world_object, T_forearm_manipulator, T_manipulator_flange):
    """Computes the target FLANGE pose in world frame required for reaching_service."""
    T_object_forearm    = reconstruct_T_object_forearm(q)
    T_world_forearm     = T_world_object @ T_object_forearm
    T_world_manipulator = T_world_forearm @ T_forearm_manipulator
    
    # Transform palm/manipulator pose to flange pose
    T_world_flange = T_world_manipulator @ T_manipulator_flange

    xyz_flange, quat_flange = matrix_to_xyz_quat(T_world_flange)
    return xyz_flange, quat_flange, q[6:30]


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

        # Compute T_manipulator_flange (Inverse of T_flange_manipulator)
        T_flange_manipulator = xyz_rpy_to_matrix(
            T_FLANGE_TO_MANIPULATOR_XYZ,
            T_FLANGE_TO_MANIPULATOR_RPY
        )
        self.T_manipulator_flange = tft.inverse_matrix(T_flange_manipulator)

        rospy.init_node("dro_arm_executor", anonymous=False)

        # Live TF lookup for rh_forearm -> rh_palm (crosses rh_WRJ2/rh_WRJ1,
        # so it is only valid at a specific, actually-measured wrist config).
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.loginfo("Waiting for /arm_motion/move_to_pose service...")
        rospy.wait_for_service("/arm_motion/move_to_pose")
        self._move_arm = rospy.ServiceProxy("/arm_motion/move_to_pose", MoveToPose)

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

    def move_arm_cartesian(self, xyz, quat, velocity_scaling=0.3):
        """Helper to send a Cartesian move command (FLANGE POSE) to reaching_service."""
        req = MoveToPoseRequest()
        req.motion_mode = req.MOTION_POSE
        req.target_pose.position.x = float(xyz[0])
        req.target_pose.position.y = float(xyz[1])
        req.target_pose.position.z = float(xyz[2])
        req.target_pose.orientation.x = float(quat[0])
        req.target_pose.orientation.y = float(quat[1])
        req.target_pose.orientation.z = float(quat[2])
        req.target_pose.orientation.w = float(quat[3])
        req.velocity_scaling = velocity_scaling
        req.wait_for_confirmation = False

        try:
            resp = self._move_arm(req)
            if not resp.success:
                rospy.logwarn(f"Cartesian motion failed. Reason: {resp.reason}")
                return False
            return True
        except rospy.ServiceException as e:
            rospy.logerr(f"MoveToPose call failed: {e}")
            return False

    def lookup_T_forearm_palm(self, timeout=2.0):
        """Live TF lookup of rh_forearm -> rh_palm.

        Reflects the hand's ACTUAL measured wrist configuration right now
        (tendon friction / tracking error included), rather than an FK
        computed from commanded joint angles.
        """
        try:
            ts = self.tf_buffer.lookup_transform(
                HAND_BASE_FRAME, EE_FRAME, rospy.Time(0), rospy.Duration(timeout)
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"TF lookup {HAND_BASE_FRAME} -> {EE_FRAME} failed: {e}")
            raise

        t = ts.transform.translation
        q = ts.transform.rotation
        return xyz_quat_to_matrix(
            np.array([t.x, t.y, t.z]),
            np.array([q.x, q.y, q.z, q.w]),
        )

    def execute_grasp(self, q):
        # ---------------------------------------------------------------------
        # STEP 1: Place all hand joints to preshape
        # ---------------------------------------------------------------------
        rospy.loginfo("Step 1: Setting hand joints to preshape...")
        medium_wrap_preshape = np.zeros(24)
        medium_wrap_preshape[20] = 1.2
        self.publish_hand_joints(medium_wrap_preshape)
        rospy.sleep(1.0)  # Short pause to let preshape complete

        # ---------------------------------------------------------------------
        # STEP 2: Reach the target FLANGE pose for the arm
        # ---------------------------------------------------------------------
        xyz_flange, quat_flange, _ = dro_q_to_world_flange(
            q, self.T_world_obj, self.T_forearm_manipulator, self.T_manipulator_flange
        )

        rospy.loginfo(f"Step 2: Reaching target arm flange pose...\n  xyz: {np.round(xyz_flange, 4)}\n  quat: {np.round(quat_flange, 4)}")
        if not self.move_arm_cartesian(xyz_flange, quat_flange, velocity_scaling=0.1):
            rospy.logerr("Failed to reach target arm pose.")
            return False

        # ---------------------------------------------------------------------
        # STEP 3: Outer grasp and 10s wait for user placement
        # ---------------------------------------------------------------------
        _, _, joints_outer = dro_q_to_world_flange(
            self.grasp_outer, self.T_world_obj, self.T_forearm_manipulator, self.T_manipulator_flange
        )
        rospy.loginfo("  Executing outer grasp...")
        self.publish_hand_joints(joints_outer)
        rospy.loginfo("Step 3: Waiting 10 seconds for user to place the object...")
        rospy.sleep(5.0)

        # ---------------------------------------------------------------------
        # STEP 4: Apply controlled offset on the arm position, along rh_palm axes
        # ---------------------------------------------------------------------
        rospy.loginfo(f"Step 4: Applying configured offset in {EE_FRAME} frame:\n"
                       f"  XYZ Delta: {OFFSET_XYZ}\n  RPY Delta (rad): {OFFSET_RPY}")

        # The hand is currently holding the OUTER-grasp preshape (published in
        # Step 3, settled for 5s), so read the wrist transform LIVE off tf --
        # this is the real, measured rh_forearm -> rh_palm pose right now.
        T_forearm_palm = self.lookup_T_forearm_palm()

        T_object_forearm = reconstruct_T_object_forearm(self.grasp)
        T_world_forearm  = self.T_world_obj @ T_object_forearm
        T_world_palm     = T_world_forearm @ T_forearm_palm
        palm_xyz, palm_quat = matrix_to_xyz_quat(T_world_palm)

        # Perturb in the palm's own frame (translation AND rotation are both
        # expressed along rh_palm's local axes).
        offset_palm_xyz, offset_palm_quat = apply_local_offset(
            palm_xyz, palm_quat, OFFSET_XYZ, OFFSET_RPY
        )

        # Map the perturbed palm pose back down to a flange target:
        # world_palm_offset -> world_forearm_offset -> world_flange_offset
        T_world_palm_offset    = xyz_quat_to_matrix(offset_palm_xyz, offset_palm_quat)
        T_world_forearm_offset = T_world_palm_offset @ tft.inverse_matrix(T_forearm_palm)
        T_world_flange_offset  = T_world_forearm_offset @ self.T_forearm_manipulator @ self.T_manipulator_flange

        offset_xyz, offset_quat = matrix_to_xyz_quat(T_world_flange_offset)
        rospy.loginfo(f"  Target Offset xyz: {np.round(offset_xyz, 4)}\n  Target Offset quat: {np.round(offset_quat, 4)}")

        if not self.move_arm_cartesian(offset_xyz, offset_quat, velocity_scaling=0.1):
            rospy.logerr("Failed to apply offset arm motion.")
            return False
        
        rospy.sleep(1.5)

        # ---------------------------------------------------------------------
        # STEP 5: Sequential grasps (Mid -> Inner)
        # ---------------------------------------------------------------------
        rospy.loginfo("Step 5: Executing sequential grasps (Mid -> Inner)...")
        _, _, joints_mid   = dro_q_to_world_flange(self.grasp,       self.T_world_obj, self.T_forearm_manipulator, self.T_manipulator_flange)
        _, _, joints_inner = dro_q_to_world_flange(self.grasp_inner, self.T_world_obj, self.T_forearm_manipulator, self.T_manipulator_flange)

        # Grasp 2: Mid
        rospy.loginfo("  Executing mid grasp...")
        self.publish_hand_joints(joints_mid)
        rospy.sleep(1.5)

        # Grasp 3: Inner
        rospy.loginfo("  Executing inner grasp...")
        self.publish_hand_joints(joints_inner)
        rospy.sleep(1.5)

        # ---------------------------------------------------------------------
        # STEP 6: Lift object by moving current Cartesian pose +15cm upwards
        # ---------------------------------------------------------------------
        rospy.loginfo("Step 6: Lifting object +15cm vertically in Cartesian space...")
        
        # Take the current (or last targeted) flange position and shift Z by +0.15m
        lift_xyz = offset_xyz.copy()
        lift_xyz[2] += 0.15
        
        if not self.move_arm_cartesian(lift_xyz, offset_quat, velocity_scaling=0.1):
            rospy.logerr("Failed to execute upward Cartesian lift motion.")
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
# Diagnostic: print the reconstructed pose for a given q
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

    rospack = rospkg.RosPack()
    package_path = rospack.get_path("my_package")
    predicted_grasps_path = os.path.join(package_path, "data", "predicted_grasps.npy")
    all_grasps = np.load(predicted_grasps_path)
    print(f"DEBUG - Array shape: {all_grasps.shape}, dtype: {all_grasps.dtype}")
    
    grasp_index = 13
    grasp_index = grasp_index - 1  # 1-indexed to 0-indexed


    grasps = all_grasps[grasp_index]
    grasp       = grasps[0]
    grasp_outer = grasps[1]
    grasp_inner = grasps[2]

    print_reconstruction_diagnostic(grasp)

    object_xyz = [1.279, 0.16, 0.74 + 0.085]  # add cylinder height to center the xyz
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