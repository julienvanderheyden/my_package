#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose
from copy import deepcopy
import tf.transformations as tf

from my_package.srv import MoveCartesian, MoveCartesianRequest, MoveCartesianResponse
from my_package.srv import GetPose, GetPoseRequest, GetPoseResponse


class ArmController:
    def __init__(self, group_name="right_arm"):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("arm_controller", anonymous=True)
        
        self.move_group = moveit_commander.MoveGroupCommander(group_name)
        self.planning_frame = self.move_group.get_planning_frame()
        self.eef_link = self.move_group.get_end_effector_link()
        
        # Service to move to a Cartesian pose
        self.move_service = rospy.Service(
            '/arm/move_cartesian', 
            MoveCartesian, 
            self.move_cartesian_callback
        )
        
        # Service to get current pose
        self.get_pose_service = rospy.Service(
            '/arm/get_current_pose',
            GetPose,
            self.get_current_pose_callback
        )
        
        rospy.loginfo(f"Arm Controller initialized")
        rospy.loginfo(f"Planning frame: {self.planning_frame}")
        rospy.loginfo(f"End effector link: {self.eef_link}")

    def get_current_pose(self):
        """Return the current end-effector pose."""
        return self.move_group.get_current_pose(self.eef_link).pose

    def reach_cartesian(self, position, orientation_quat, eef_step=0.005, speed_factor=1.0):
        """
        Move end-effector to target position using Cartesian path.
        Args:
            position: [x, y, z] in meters
            orientation_quat: [x, y, z, w] quaternion
            eef_step: distance between waypoints
            speed_factor: speed multiplier (1.0 = normal speed)
        Returns:
            (success, fraction_complete)
        """
        # Fixed transform (ra_flange -> rh_palm)
        t_flange_palm = [0.247, 0.000, 0.010]
        r_flange_palm = [-1.575, 0.000, -1.563]  # Euler angles for fixed transform
        T_flange_palm = tf.euler_matrix(*r_flange_palm)
        T_flange_palm[0:3, 3] = t_flange_palm
        
        # Compute inverse transform
        T_palm_flange = tf.inverse_matrix(T_flange_palm)
        
        # Desired palm pose (using quaternion)
        T_base_palm = tf.quaternion_matrix(orientation_quat)
        T_base_palm[0:3, 3] = position
        
        # Compute flange pose
        T_base_flange = T_base_palm @ T_palm_flange
        flange_position = tf.translation_from_matrix(T_base_flange)
        flange_quat = tf.quaternion_from_matrix(T_base_flange)
        
        # Create target pose
        current_pose = deepcopy(self.get_current_pose())
        target_pose = deepcopy(current_pose)
        
        target_pose.position.x = flange_position[0]
        target_pose.position.y = flange_position[1]
        target_pose.position.z = flange_position[2]
        
        target_pose.orientation.x = flange_quat[0]
        target_pose.orientation.y = flange_quat[1]
        target_pose.orientation.z = flange_quat[2]
        target_pose.orientation.w = flange_quat[3]
        
        waypoints = [target_pose]
        
        # Plan Cartesian path
        (plan, fraction) = self.move_group.compute_cartesian_path(
            waypoints,
            eef_step=eef_step,
            jump_threshold=0.0
        )
        
        if fraction < 0.95:
            rospy.logwarn(f"Cartesian path only {fraction*100:.1f}% complete!")
            return False, fraction
        
        # Adjust speed
        for point in plan.joint_trajectory.points:
            point.time_from_start *= speed_factor
            if speed_factor != 1.0:
                point.velocities = [v / speed_factor for v in point.velocities]
                point.accelerations = [a / (speed_factor**2) for a in point.accelerations]
        
        # Execute
        self.move_group.execute(plan, wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        
        return True, fraction

    def move_cartesian_callback(self, req):
        """Service callback to move to a Cartesian pose."""
        position = [req.position.x, req.position.y, req.position.z]
        orientation_quat = [
            req.orientation.x, 
            req.orientation.y, 
            req.orientation.z,
            req.orientation.w
        ]
        
        rospy.loginfo(f"Moving to position: {position}, orientation (quat): {orientation_quat}")
        
        success, fraction = self.reach_cartesian(
            position, 
            orientation_quat, 
            eef_step=req.eef_step if req.eef_step > 0 else 0.005,
            speed_factor=req.speed_factor if req.speed_factor > 0 else 1.0
        )
        
        response = MoveCartesianResponse()
        response.success = success
        response.fraction_complete = fraction
        response.message = "Motion completed" if success else "Motion failed"
        
        return response

    def get_current_pose_callback(self, req):
        """Service callback to get current pose."""
        current_pose = self.get_current_pose()
        
        response = GetPoseResponse()
        response.pose = current_pose
        
        return response

    def shutdown(self):
        moveit_commander.roscpp_shutdown()
        rospy.loginfo("Arm Controller shut down.")


if __name__ == "__main__":
    try:
        controller = ArmController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        if 'controller' in locals():
            controller.shutdown()