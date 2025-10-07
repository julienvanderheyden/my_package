#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
import geometry_msgs.msg
from math import pi
from tf.transformations import quaternion_from_euler


class UR10eMoveItController:
    def __init__(self, group_name="right_arm"):
        # Initialize MoveIt Commander and ROS node (if not already done)
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("ur10e_cartesian_goal", anonymous=True)

        # Initialize Move Group
        self.move_group = moveit_commander.MoveGroupCommander(group_name)

        # Log useful info
        self.planning_frame = self.move_group.get_planning_frame()
        self.eef_link = self.move_group.get_end_effector_link()

        rospy.loginfo(f"Planning frame: {self.planning_frame}")
        rospy.loginfo(f"End effector link: {self.eef_link}")

    def reach(self, position, orientation):
        """
        Plans and executes a trajectory to reach a given pose.

        Args:
            position: (x, y, z) target position in meters
            orientation: (roll, pitch, yaw) orientation in radians
            execute: if True, execute the planned trajectory
        """

        # Define target pose
        target_pose = geometry_msgs.msg.Pose()
        target_pose.position.x = position[0]
        target_pose.position.y = position[1]
        target_pose.position.z = position[2]

        roll, pitch, yaw = orientation
        q = quaternion_from_euler(roll, pitch, yaw)
        target_pose.orientation.x = q[0]
        target_pose.orientation.y = q[1]
        target_pose.orientation.z = q[2]
        target_pose.orientation.w = q[3]

        # Set the pose target
        self.move_group.set_pose_target(target_pose, self.eef_link)

        rospy.loginfo("Planning trajectory to target pose...")
        plan_result = self.move_group.plan()

        # Handle the tuple output (depending on MoveIt version)
        if isinstance(plan_result, tuple):
            plan = plan_result[1] if len(plan_result) > 1 else plan_result[0]
        else:
            plan = plan_result

        if not hasattr(plan, "joint_trajectory") or len(plan.joint_trajectory.points) == 0:
            rospy.logerr("Planning failed!")
            return False

        rospy.loginfo("Planning successful!")

        rospy.loginfo("Executing trajectory...")
        self.move_group.execute(plan, wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        rospy.loginfo("Motion complete.")

        return True

    def get_current_pose(self):
        """Return the current end-effector pose (geometry_msgs/Pose)."""
        return self.move_group.get_current_pose(self.eef_link).pose

    def shutdown(self):
        """Cleanly shutdown MoveIt Commander."""
        moveit_commander.roscpp_shutdown()
        rospy.loginfo("MoveIt Commander shut down.")


if __name__ == "__main__":
    controller = UR10eMoveItController()

    # Example usage
    position = [0.724, 0.173, 1.07]
    orientation = [1.5 * pi, 0, 0]

    controller.reach(position, orientation)

    # Print current pose
    current_pose = controller.get_current_pose()
    rospy.loginfo(f"Current EE pose:\n{current_pose}")

    controller.shutdown()
