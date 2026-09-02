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

from geometry_msgs.msg import Pose
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import JointState

from my_package.srv import MoveToPose, MoveToPoseRequest

# ═══════════════════════════════════════════════════════════════════════════════
# Constant: rh_forearm → rh_manipulator  (from URDF / live tf)
# ═══════════════════════════════════════════════════════════════════════════════

T_FOREARM_TO_MANIPULATOR_XYZ  = np.array([0.001, -0.002, 0.296])
T_FOREARM_TO_MANIPULATOR_QUAT = np.array([-0.077, 0.003, 0.0, 0.997])  # [x,y,z,w]


# ═══════════════════════════════════════════════════════════════════════════════
# Manual Perturbation / Offset Parameters (Editable)
# ═══════════════════════════════════════════════════════════════════════════════

# Specify exact translation offsets [dx, dy, dz] in meters
OFFSET_XYZ = np.array([0.00, 0.00, 0.000])  

# Specify exact rotation offsets [roll, pitch, yaw] in radians (or use np.radians(deg))
OFFSET_RPY = np.array([np.radians(0.0), np.radians(0.0), np.radians(0.0)]) 


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
    offset_xyz = xyz + delta_xyz

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

    def move_arm_cartesian(self, xyz, quat, velocity_scaling= 0.3):
        """Helper to send a Cartesian move command to reaching_service."""
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

    def move_arm_joint(self, joint_values):
        """Helper to send a Joint Space move command to reaching_service."""
        req = MoveToPoseRequest()
        req.motion_mode = req.MOTION_JOINT
        req.ra_elbow_joint = float(joint_values["ra_elbow_joint"])
        req.ra_shoulder_lift_joint = float(joint_values["ra_shoulder_lift_joint"])
        req.ra_shoulder_pan_joint = float(joint_values["ra_shoulder_pan_joint"])
        req.ra_wrist_1_joint = float(joint_values["ra_wrist_1_joint"])
        req.ra_wrist_2_joint = float(joint_values["ra_wrist_2_joint"])
        req.ra_wrist_3_joint = float(joint_values["ra_wrist_3_joint"])
        req.velocity_scaling = 0.5
        req.wait_for_confirmation = False

        try:
            resp = self._move_arm(req)
            if not resp.success:
                rospy.logwarn(f"Joint motion failed. Reason: {resp.reason}")
                return False
            return True
        except rospy.ServiceException as e:
            rospy.logerr(f"MoveToPose joint move failed: {e}")
            return False

    def execute_grasp(self, q):
        # ---------------------------------------------------------------------
        # STEP 1: Place all hand joints to zero for preshaping
        # ---------------------------------------------------------------------
        rospy.loginfo("Step 1: Setting hand joints to zero for preshaping...")
        zero_joints = np.zeros(24)
        medium_wrap_preshape = np.zeros(24)
        medium_wrap_preshape[20] = 1.2
        self.publish_hand_joints(medium_wrap_preshape)
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
        # STEP 3: dro preshape and waits 10 seconds for user placement
        # ---------------------------------------------------------------------
        _, _, joints_outer = dro_q_to_world_manipulator(self.grasp_outer, self.T_world_obj, self.T_forearm_manipulator)
        # Grasp 1: Outer
        rospy.loginfo("  Executing outer grasp...")
        self.publish_hand_joints(joints_outer)
        rospy.loginfo("Step 3: Waiting 10 seconds for user to place the object...")
        rospy.sleep(10.0)

        # ---------------------------------------------------------------------
        # STEP 4: Apply controlled deterministic perturbation on the arm position
        # ---------------------------------------------------------------------
        rospy.loginfo(f"Step 4: Applying configured offset:\n  XYZ Delta: {OFFSET_XYZ}\n  RPY Delta (rad): {OFFSET_RPY}")
        offset_xyz, offset_quat = apply_offset(xyz, quat, OFFSET_XYZ, OFFSET_RPY)
        rospy.loginfo(f"  Target Offset xyz: {np.round(offset_xyz, 4)}\n  Target Offset quat: {np.round(offset_quat, 4)}")

        if not self.move_arm_cartesian(offset_xyz, offset_quat, velocity_scaling=0.1):
            rospy.logerr("Failed to apply offset arm motion.")
            return False
        
        rospy.sleep(1.5)

        # ---------------------------------------------------------------------
        # STEP 5: Run three grasp sequences with 1.5s interval
        # ---------------------------------------------------------------------
        rospy.loginfo("Step 5: Executing sequential grasps (Outer -> Mid -> Inner)...")
        
        
        _, _, joints_mid   = dro_q_to_world_manipulator(self.grasp,       self.T_world_obj, self.T_forearm_manipulator)
        _, _, joints_inner = dro_q_to_world_manipulator(self.grasp_inner, self.T_world_obj, self.T_forearm_manipulator)



        # Grasp 2: Mid
        rospy.loginfo("  Executing mid grasp...")
        self.publish_hand_joints(joints_mid)
        rospy.sleep(1.5)

        # Grasp 3: Inner
        rospy.loginfo("  Executing inner grasp...")
        self.publish_hand_joints(joints_inner)
        rospy.sleep(1.5)

        # ---------------------------------------------------------------------
        # STEP 6: Lift the object using explicit Joint Space targets
        # ---------------------------------------------------------------------
        rospy.loginfo("Step 6: Lifting object using joint space targets...")
        try:
            # Get current joint states to preserve ra_wrist_3_joint angle
            joint_state_msg = rospy.wait_for_message("/joint_states", JointState, timeout=5.0)
            current_wrist_3 = 0.0
            if "ra_wrist_3_joint" in joint_state_msg.name:
                idx = joint_state_msg.name.index("ra_wrist_3_joint")
                current_wrist_3 = joint_state_msg.position[idx]

            lift_joint_values = {
                "ra_elbow_joint": 1.6,
                "ra_shoulder_lift_joint": -1.0,
                "ra_shoulder_pan_joint": -0.09,
                "ra_wrist_1_joint": -0.585,
                "ra_wrist_2_joint": 1.48,
                "ra_wrist_3_joint": current_wrist_3,
            }

            if not self.move_arm_joint(lift_joint_values):
                rospy.logerr("Failed to execute joint lift motion.")
                return False

        except rospy.ROSException as e:
            rospy.logerr(f"Failed to read current joint states for lift: {e}")
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

    test_cylinder_5cm = np.array([[0.0741, 0.3093, 0.1152, 1.5283, -0.4392, 0.2012, 0.0456, 0.3773, -0.3491, 0.6304, 0.2911, 0.0, -0.2684, 0.255, 1.261, -0.0, -0.3117, 0.4097, 1.4692, 0.6834, 0.2742, -0.3491, 0.9381, 1.2706, 1.1324, -0.4138, 0.8833, 0.1824, -0.099, 1.0868], [0.0741, 0.3093, 0.1152, 1.5283, -0.4392, 0.2012, 0.0456, 0.3773, -0.3491, 0.4073, 0.2184, 0.0, -0.114, 0.1258, 0.9458, -0.0, -0.321, 0.2418, 1.1019, 0.9053, 0.2057, -0.3491, 0.6381, 1.3456, 1.242, -0.5721, 0.9679, 0.1892, -0.2488, 0.7496], [0.0741, 0.3093, 0.1152, 1.5283, -0.4392, 0.2012, 0.0456, 0.3773, -0.2444, 0.7714, 0.4831, 0.2356, -0.2805, 0.4524, 1.3075, 0.2356, -0.2126, 0.5838, 1.4845, 0.5809, 0.3509, -0.2444, 1.033, 1.08, 0.9625, -0.1946, 0.7508, 0.1236, 0.0206, 1.1594]])
    
    grasps = test_cylinder_5cm

    # rospack = rospkg.RosPack()
    # package_path = rospack.get_path("my_package")
    # predicted_grasps_path = os.path.join(package_path, "data", "predicted_grasps.npy")
    # all_grasps = np.load(predicted_grasps_path)
    # print(f"DEBUG - Array shape: {all_grasps.shape}, dtype: {all_grasps.dtype}")
    # grasp_index = 14
    # grasp_index = grasp_index -1 #for python indexing 
    # grasps = all_grasps[grasp_index]

    grasp = grasps[0]
    grasp_outer = grasps[1]
    grasp_inner = grasps[2]

    print_reconstruction_diagnostic(grasp)

    object_xyz = [1.279, 0.14, 0.74+0.085] # add cylinder height to center the xyz
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