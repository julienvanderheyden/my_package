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

        #MOVING TO MID GRASP POSITION
        rospy.sleep(10.0)
        joints_cmd = [float(joints[i]) for i in REINDEX]
        msg.data = joints_cmd
        self._joint_pub.publish(msg)

        # MOVING TO INNER GRASP POSITION
        rospy.sleep(1.5)  # wait for the first command to take effect
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

    grasps_8cm = np.array([[-0.31280833,  0.06642898, -0.08719411,  0.99932575,  1.0106376 ,
        1.7023215 , -0.215437  ,  0.4084922 ,  0.18576908,  0.16372374,
        0.6031984 ,  0.8053129 , -0.09290206,  0.31394938,  0.5226503 ,
        0.8624907 ,  0.24004082,  0.69580424,  0.6150156 ,  0.14475593,
        0.74541855, -0.13865   ,  0.69575006,  0.1839977 ,  1.4259698 ,
        0.44490644,  1.221874  ,  0.20948908, -0.6980942 , -0.26179644],
        [-0.31280833,  0.06642898, -0.08719411,  0.99932575,  1.0106376 ,
        1.7023215 , -0.215437  ,  0.4084922 ,  0.22659327,  0.05734295,
        0.4523988 ,  0.60398465,  0.01758991,  0.17001218,  0.39198774,
        0.64686805,  0.09276415,  0.45640332,  0.46126172,  0.10856695,
        0.5590639 , -0.19125396,  0.4563627 ,  0.13799828,  1.0694773 ,
        0.07188043,  1.2218381 ,  0.20947668, -0.69810355, -0.2617972 ],
        [-0.31280833,  0.06642898, -0.08719411,  0.99932575,  1.0106376 ,
        1.7023215 , -0.215437  ,  0.4084922 ,  0.10554384,  0.37478465,
        0.7483381 ,  0.9201354 , -0.13132663,  0.50247645,  0.6798722 ,
        0.9687366 ,  0.25639457,  0.82705307,  0.75838274,  0.358662  ,
        0.7514155 , -0.06549262,  0.827007  ,  0.3920175 ,  1.4476938 ,
        0.5352501 ,  1.0385929 ,  0.14664978, -0.48866028,  0.01309249],
    ], dtype=float)

    grasps_2cm = np.array([[-3.6556e-01, -5.6917e-02,  6.7579e-02,  3.0456e+00,  1.5044e+00,
         3.8788e-01, -1.9844e-01, -1.6277e-01,  3.4909e-01,  7.2222e-01,
         1.6983e-01,  1.5708e+00,  1.3004e-01,  8.4060e-01, -1.7798e-05,
         1.5708e+00, -6.3998e-03,  9.1116e-01,  2.3698e-01,  3.8181e-06,
         1.6861e-01,  4.4037e-02,  8.3108e-01,  1.3096e-01,  1.5708e+00,
         3.9912e-01,  1.0990e+00,  2.0943e-01,  6.9812e-01, -2.6179e-01],
         [-3.6556e-01, -5.6917e-02,  6.7579e-02,  3.0456e+00,  1.5044e+00,
         3.8788e-01, -1.9844e-01, -1.6277e-01,  1.7455e-01,  4.7622e-01,
         1.2737e-01,  1.5708e+00,  1.0263e-02,  5.6500e-01, -1.3349e-05,
         1.1781e+00,  8.2467e-02,  6.1792e-01,  1.7773e-01,  2.8636e-06,
         1.2646e-01,  1.2029e-01,  5.5786e-01,  4.9092e-01,  1.1781e+00,
         3.7542e-02,  8.2424e-01,  1.0472e-01,  3.4905e-01, -2.6179e-01],
         [-0.3656, -0.0569,  0.0676,  3.0456,  1.5044,  0.3879, -0.1984, -0.1628,
         0.3491,  0.8495,  0.3800,  1.3352,  0.1629,  0.9501,  0.2356,  1.5708,
        -0.0578,  1.0101,  0.4370,  0.2356,  0.2611, -0.0149,  0.9420,  0.1113,
         1.5708,  0.4963,  1.1174,  0.2094,  0.6981,  0.0131]], dtype=float)
    
    grasps_plate1cm = np.array([[-0.2712116241455078, -0.13351529836654663, 0.2013239860534668, 
    -2.8174047470092773, 0.8478065729141235, -0.32263267040252686, 0.056119367480278015, -0.3466302156448364, 
    -0.05283015966415405, 0.9725329279899597, 0.17835502326488495, 0.9705254435539246, -0.2105221003293991, 
    0.9871237874031067, 0.10741354525089264, 0.9516901969909668, 0.056143149733543396, 1.115608811378479, 
    0.23362839221954346, 0.6929372549057007, 0.31713002920150757, -0.15575332939624786, 1.1002352237701416, 
    0.09513037651777267, 1.0373584032058716, 0.37452876567840576, 1.2217332124710083, 0.06441983580589294, 
    -0.5752281546592712, 0.9932101964950562], 
    [-0.2712116241455078, -0.13351529836654663, 0.2013239860534668, -2.8174047470092773, 
     0.8478065729141235, -0.32263267040252686, 0.056119367480278015, -0.3466302156448364, -0.12688907980918884, 
     0.6639498472213745, 0.1337662637233734, 0.7278940677642822, -0.07062511146068573, 0.6748930215835571, 
     0.08056016266345978, 0.7137676477432251, -0.045159101486206055, 0.7712567448616028, 0.1752212941646576, 
     0.5197029113769531, 0.23784752190113068, -0.2040814608335495, 0.7597265839576721, 0.07134778052568436, 
     0.7780188322067261, 0.019097179174423218, 0.9162999391555786, 0.10067475587129593, -0.6059540510177612, 0.6794577836990356], 
     [-0.2712116241455078, -0.13351529836654663, 0.2013239860534668, -2.8174047470092773, 
      0.8478065729141235, -0.32263267040252686, 0.056119367480278015, -0.3466302156448364, 0.007454242557287216, 
      1.0622724294662476, 0.38722124695777893, 1.0605660676956177, -0.23130366206169128, 1.0746747255325317, 
      0.3269209861755371, 1.0445561408996582, 0.10008154809474945, 1.183887004852295, 0.4342035949230194, 
      0.8246161341667175, 0.3873702585697174, -0.08003045618534088, 1.1708194017410278, 0.31648027896881104, 
      1.117374062538147, 0.47542908787727356, 1.2217328548431396, 0.02334093302488327, -0.38422417640686035, 1.0798481702804565]])
    
    grasps = grasps_plate1cm

    grasp = grasps[0]
    grasp_outer = grasps[1]
    grasp_inner = grasps[2]

    # Run diagnostic to inspect the rotation reconstruction
    print_reconstruction_diagnostic(grasp)

    #object_xyz = [1.25, 0.215, 0.84]
    object_xyz = [1.25, 0.215, 0.74+0.23]
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