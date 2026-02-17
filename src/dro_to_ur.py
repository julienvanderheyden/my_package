#!/usr/bin/env python3
"""
dro_to_ur.py
============
Bridges D(R,O) Grasp inference output to the real ShadowHand + arm.

═══════════════════════════════════════════════════════════════════════════════
HOW THE VIRTUAL JOINTS ENCODE ROTATION — THE KEY ISSUE
═══════════════════════════════════════════════════════════════════════════════

The DRO ShadowHand URDF has 6 virtual joints at its root:
    virtual_x   → prismatic along X  →  q[0]  (translation, metres)
    virtual_y   → prismatic along Y  →  q[1]  (translation, metres)
    virtual_z   → prismatic along Z  →  q[2]  (translation, metres)
    virtual_rx  → revolute about X   →  q[3]  (angle, radians)
    virtual_ry  → revolute about Y   →  q[4]  (angle, radians)
    virtual_rz  → revolute about Z   →  q[5]  (angle, radians)

Because these are processed by pytorch_kinematics as a URDF kinematic chain,
each revolute value is a single scalar angle about its OWN fixed axis.
They are NOT Euler angles fed simultaneously into euler_matrix().

The correct rotation matrix is the chain product:
    R = Rx(q[3]) · Ry(q[4]) · Rz(q[5])

where each Ri is a basic rotation matrix about that axis.

This is the standard URDF revolute-joint composition: each joint rotates
about the axis specified in the URDF <axis> element, applied in sequence
along the kinematic chain.

Treating q[3:6] as (roll, pitch, yaw) passed to euler_matrix() happens to
give the same formula ONLY if the axes are exactly X, Y, Z in that order —
but the convention (intrinsic vs extrinsic, order) differs from scipy/tf,
causing sign and ordering errors for non-trivial orientations.

═══════════════════════════════════════════════════════════════════════════════
FULL TRANSFORM CHAIN
═══════════════════════════════════════════════════════════════════════════════

    T_world_manipulator =
        T_world_object            ← known object pose in robot world frame
      · T_object_forearm          ← reconstructed from q[0:6] (see above)
      · T_forearm_manipulator     ← fixed URDF offset (fill in below)

The result goes directly to arm_motion_service, which commands rh_manipulator.

═══════════════════════════════════════════════════════════════════════════════
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
# Run:  rosrun tf tf_echo rh_forearm rh_manipulator
# ═══════════════════════════════════════════════════════════════════════════════

T_FOREARM_TO_MANIPULATOR_XYZ  = np.array([0.001, -0.002, 0.296])
T_FOREARM_TO_MANIPULATOR_QUAT = np.array([-0.077, 0.003, 0.0, 0.997])  # [x,y,z,w]


# ═══════════════════════════════════════════════════════════════════════════════
# Finger joint names
# ═══════════════════════════════════════════════════════════════════════════════

# Order in which DRO outputs q[6:30]  (matches the DRO ShadowHand URDF chain)
DRO_JOINT_ORDER = [
    "rh_WRJ2", "rh_WRJ1",
    "rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_FFJ1",
    "rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1",
    "rh_RFJ4", "rh_RFJ3", "rh_RFJ2", "rh_RFJ1",
    "rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2", "rh_LFJ1",
    "rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1",
]

# Order expected by the hand command topic
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

# Precompute: for each position in CMD_JOINT_ORDER, which index in DRO_JOINT_ORDER?
# i.e. reordered[i] = dro_joints[REINDEX[i]]
_dro_pos = {name: i for i, name in enumerate(DRO_JOINT_ORDER)}
REINDEX = [_dro_pos[name] for name in CMD_JOINT_ORDER]


# ═══════════════════════════════════════════════════════════════════════════════
# Math helpers
# ═══════════════════════════════════════════════════════════════════════════════

def Rx(a):
    """4×4 rotation matrix about X by angle a [rad]."""
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1,  0,   0,  0],
                     [0,  ca, -sa, 0],
                     [0,  sa,  ca, 0],
                     [0,  0,   0,  1]], dtype=float)

def Ry(a):
    """4×4 rotation matrix about Y by angle a [rad]."""
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ ca, 0, sa, 0],
                     [  0, 1,  0, 0],
                     [-sa, 0, ca, 0],
                     [  0, 0,  0, 1]], dtype=float)

def Rz(a):
    """4×4 rotation matrix about Z by angle a [rad]."""
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0, 0],
                     [sa,  ca, 0, 0],
                     [ 0,   0, 1, 0],
                     [ 0,   0, 0, 1]], dtype=float)

def xyz_quat_to_matrix(xyz, quat):
    """4×4 matrix from translation + quaternion [x, y, z, w]."""
    T = tft.quaternion_matrix(quat)
    T[0:3, 3] = xyz
    return T

def xyz_rpy_to_matrix(xyz, rpy):
    """4×4 matrix from translation + extrinsic RPY [rad]."""
    T = tft.euler_matrix(rpy[0], rpy[1], rpy[2], axes='sxyz')
    T[0:3, 3] = xyz
    return T

def matrix_to_xyz_quat(T):
    """Extract (xyz, quat [x,y,z,w]) from 4×4 matrix."""
    return T[0:3, 3].copy(), tft.quaternion_from_matrix(T)


# ═══════════════════════════════════════════════════════════════════════════════
# Core: reconstruct T_object_forearm from q[0:6]
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruct_T_object_forearm(q):
    """
    Reconstruct the rh_forearm pose in the object frame from q[0:6].

    The DRO virtual joints are a kinematic chain of 6 joints processed by
    pytorch_kinematics in URDF order.  Their effect compounds as:

        T = Trans(q[0], q[1], q[2]) · Rx(q[3]) · Ry(q[4]) · Rz(q[5])

    where each step is the standard URDF joint transform for a prismatic or
    revolute joint about a fixed axis.

    NOTE: this assumes the standard DRO virtual joint URDF structure:
        virtual_x  → prismatic, axis="1 0 0"
        virtual_y  → prismatic, axis="0 1 0"
        virtual_z  → prismatic, axis="0 0 1"
        virtual_rx → revolute,  axis="1 0 0"
        virtual_ry → revolute,  axis="0 1 0"
        virtual_rz → revolute,  axis="0 0 1"
    applied in that order in the URDF chain.

    If your DRO URDF has a different axis order or different axes, adjust
    the composition below accordingly.
    """
    # Translation part (prismatic joints are purely additive)
    T_trans = np.eye(4)
    T_trans[0, 3] = q[0]  # virtual_x
    T_trans[1, 3] = q[1]  # virtual_y
    T_trans[2, 3] = q[2]  # virtual_z

    # Rotation part: chain of three single-axis rotations IN URDF ORDER
    T_rot = Rx(q[3]) @ Ry(q[4]) @ Rz(q[5])

    return T_trans @ T_rot


def dro_q_to_world_manipulator(q, T_world_object, T_forearm_manipulator):
    """
    Full pipeline: DRO q vector → desired rh_manipulator pose in world frame.

    Args:
        q                     : (30,) numpy array from predict_grasp()
        T_world_object         : (4,4) known pose of object centroid in world
        T_forearm_manipulator  : (4,4) fixed URDF offset rh_forearm→rh_manipulator

    Returns:
        xyz    : (3,) target position  of rh_manipulator in world [m]
        quat   : (4,) target quaternion [x,y,z,w] of rh_manipulator in world
        joints : (24,) finger + wrist joint values [rad]
    """
    T_object_forearm    = reconstruct_T_object_forearm(q)
    T_world_forearm     = T_world_object @ T_object_forearm
    T_world_manipulator = T_world_forearm @ T_forearm_manipulator

    xyz, quat = matrix_to_xyz_quat(T_world_manipulator)
    return xyz, quat, q[6:30]


# ═══════════════════════════════════════════════════════════════════════════════
# ROS executor
# ═══════════════════════════════════════════════════════════════════════════════

class DROArmExecutor:

    def __init__(self, predicted_grasp, grasp_outer, grasp_inner, T_world_object):
        self.grasp       = predicted_grasp   # (30,)
        self.grasp_outer = grasp_outer       # (30,)
        self.grasp_inner = grasp_inner       # (30,)
        self.T_world_obj = T_world_object    # (4,4)
        self.grasp_

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

        rospy.Service("/dro/execute_grasp", Trigger, self._cb_execute_grasp)
        rospy.loginfo("DROArmExecutor ready.")

    def execute_grasp(self, q):
        xyz, quat, joints = dro_q_to_world_manipulator(
            q, self.T_world_obj, self.T_forearm_manipulator
        )

        _ ,_ , joints_inner = dro_q_to_world_manipulator(
            self.grasp_inner, self.T_world_obj, self.T_forearm_manipulator
        )

        _ , _ , joints_outer = dro_q_to_world_manipulator(
            self.grasp_outer, self.T_world_obj, self.T_forearm_manipulator
        )

        rospy.loginfo(
            f"Executing grasp:\n"
            f"  rh_manipulator xyz  (world): {np.round(xyz, 4)}\n"
            f"  rh_manipulator quat (world): {np.round(quat, 4)}"
        )

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

        # MOVING TO OUTER GRASP POSITION

        # Reorder from DRO order to command topic order
        joints_cmd = [float(joints_outer[i]) for i in REINDEX]
        msg = Float64MultiArray()
        msg.data = joints_cmd
        self._joint_pub.publish(msg)

        # MOVING TO INNER GRASP POSITION
        rospy.sleep(1.0)  # wait for the first command to take effect
        joints_cmd = [float(joints_inner[i]) for i in REINDEX]
        msg.data = joints_cmd
        self._joint_pub.publish(msg)

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
    """
    Helper to sanity-check the rotation reconstruction before running on
    the real robot.  Call this standalone to inspect the result.
    """
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

    grasp = np.array([-2.8753e-01,  4.7317e-03,  2.4047e-01,  2.9475e+00,  9.5168e-01,
        -6.9413e-01, -6.5923e-02, -4.2896e-01,  2.5629e-01,  1.0210e+00,
         4.1788e-01, -4.8975e-07, -2.0706e-01,  1.1246e+00,  2.0445e-02,
         3.8373e-06,  3.4908e-01,  6.9455e-01,  1.3072e+00, -3.1413e-06,
         4.7790e-01,  1.9100e-01,  9.8466e-01,  5.9060e-02,  5.4083e-06,
         7.2397e-01,  9.9239e-01, -3.8556e-02, -5.5689e-01, -2.6180e-01], dtype=float)

    grasp_outer = np.array([-2.8753e-01,  4.7317e-03,  2.4047e-01,  2.9475e+00,  9.5168e-01,
        -6.9413e-01, -6.5923e-02, -4.2896e-01,  1.0495e-01,  7.0033e-01,
         3.1341e-01, -3.6732e-07, -6.8030e-02,  7.7802e-01,  1.5334e-02,
         2.8779e-06,  1.7454e-01,  4.5546e-01,  1.3731e+00,  3.9270e-01,
         3.5843e-01,  5.5983e-02,  6.7305e-01,  4.4295e-02,  4.0562e-06,
         8.0478e-01,  7.4429e-01, -8.1277e-02, -5.9220e-01, -2.6180e-01], dtype=float)
    
    grasp_inner = np.array([-2.8753e-01,  4.7317e-03,  2.4047e-01,  2.9475e+00,  9.5168e-01,
        -6.9413e-01, -6.5923e-02, -4.2896e-01,  2.7020e-01,  1.1035e+00,
         5.9082e-01,  2.3562e-01, -2.2836e-01,  1.1915e+00,  2.5300e-01,
         2.3562e-01,  3.4908e-01,  8.2599e-01,  1.1111e+00, -2.6701e-06,
         5.2403e-01,  2.1471e-01,  1.0726e+00,  2.8582e-01,  2.3562e-01,
         4.5830e-01,  1.0268e+00, -1.3568e-03, -3.6864e-01,  1.3091e-02], dtype=float)

    # Run diagnostic to inspect the rotation reconstruction
    print_reconstruction_diagnostic(grasp)

    object_xyz = [1.15, 0.215, 0.84]
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