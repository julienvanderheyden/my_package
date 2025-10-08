#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
import geometry_msgs.msg
from math import pi
from tf.transformations import quaternion_from_euler
from std_msgs.msg import Int32
from copy import deepcopy


class UR10eMoveItController:
    def __init__(self, group_name="right_arm"):
        # Initialize MoveIt Commander and ROS node
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("ur10e_cartesian_goal", anonymous=True)

        # Initialize Move Group
        self.move_group = moveit_commander.MoveGroupCommander(group_name)

        # Publishers and subscribers
        self.sync_sub = rospy.Subscriber(
            "/ros_julia_synchronization", Int32, self.sync_callback
        )
        self.sync_pub = rospy.Publisher(
            "/ros_julia_synchronization", Int32, queue_size=10
        )

        # Predefined positions (x, y, z) and orientations (roll, pitch, yaw)
        self.positions = [
            [0.724, 0.173, 1.07],
            [1.0, 0.173, 1.07],
            [1.0, 0.0, 1.07],
            [1.0, 0.2, 1.07],
            [1.0, 0.4, 1.07],
        ]
        self.orientations = [
            [1.5*pi, 0, 0],
            [1.5*pi, 0, 0],
            [1.5*pi, 0, 0],
            [1.5*pi, 0, 0],
            [1.5*pi, 0, 0],
        ]

        self.planning_frame = self.move_group.get_planning_frame()
        self.eef_link = self.move_group.get_end_effector_link()

        rospy.loginfo(f"Planning frame: {self.planning_frame}")
        rospy.loginfo(f"End effector link: {self.eef_link}")

        rospy.loginfo("UR10eMoveItController initialized and listening for commands...")

    # def reach(self, position, orientation):
    #     """Plan and execute trajectory to target pose."""
    #     target_pose = geometry_msgs.msg.Pose()
    #     target_pose.position.x = position[0]
    #     target_pose.position.y = position[1]
    #     target_pose.position.z = position[2]

    #     roll, pitch, yaw = orientation
    #     q = quaternion_from_euler(roll, pitch, yaw)
    #     target_pose.orientation.x = q[0]
    #     target_pose.orientation.y = q[1]
    #     target_pose.orientation.z = q[2]
    #     target_pose.orientation.w = q[3]

    #     self.move_group.set_start_state_to_current_state()
    #     self.move_group.set_pose_target(target_pose, self.eef_link)
    #     rospy.loginfo(f"Planning trajectory to pose {position}...")

    #     plan_result = self.move_group.plan()
    #     plan = (
    #         plan_result[1]
    #         if isinstance(plan_result, tuple) and len(plan_result) > 1
    #         else plan_result
    #     )

    #     if not hasattr(plan, "joint_trajectory") or len(plan.joint_trajectory.points) == 0:
    #         rospy.logerr("Planning failed!")
    #         return False

    #     rospy.loginfo("Planning successful. Executing...")
    #     self.move_group.execute(plan, wait=True)
    #     self.move_group.stop()
    #     self.move_group.clear_pose_targets()
    #     rospy.loginfo("Motion complete.")
    #     return True

    def get_current_pose(self):
        """Return the current end-effector pose (geometry_msgs/Pose)."""
        return self.move_group.get_current_pose(self.eef_link).pose

    def reach_cartesian(self, position, orientation, eef_step=0.01):
        """
        Move the end-effector to a target position using a smooth Cartesian path.

        Args:
            position: (x, y, z) in meters
            orientation: (roll, pitch, yaw) in radians
            eef_step: distance between waypoints in meters
        """
        current_pose = deepcopy(self.get_current_pose())
        target_pose = deepcopy(current_pose)

        # Set target position
        target_pose.position.x = position[0]
        target_pose.position.y = position[1]
        target_pose.position.z = position[2]

        # Set orientation
        roll, pitch, yaw = orientation
        q = quaternion_from_euler(roll, pitch, yaw)
        target_pose.orientation.x = q[0]
        target_pose.orientation.y = q[1]
        target_pose.orientation.z = q[2]
        target_pose.orientation.w = q[3]

        # Generate waypoints
        waypoints = [target_pose]

        # Compute Cartesian path
        (plan, fraction) = self.move_group.compute_cartesian_path(
            waypoints,
            eef_step=eef_step,  # e.g., 1 cm per step
            jump_threshold=0.0  # prevents IK jumps
        )

        if fraction < 1.0:
            rospy.logwarn(f"Cartesian path only {fraction*100:.1f}% complete!")

        rospy.loginfo("Executing Cartesian path...")
        self.move_group.set_max_velocity_scaling_factor(0.01)
        self.move_group.set_max_acceleration_scaling_factor(0.01)
        self.move_group.execute(plan, wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        rospy.loginfo("Motion complete.")


    def sync_callback(self, msg):
        """Callback for incoming messages from Julia."""
        command = int(msg.data)
        if 1 <= command <= 5:
            rospy.loginfo(f"Received command: {command} → moving to pose {command}")
            position = self.positions[command - 1]
            orientation = self.orientations[command - 1]
            success = self.reach_cartesian(position, orientation)
            if success:
                rospy.sleep(0.5)  # small delay before publishing back
                rospy.loginfo("Publishing 0 to indicate completion.")
                self.sync_pub.publish(Int32(0))
        elif command == 0:
            rospy.loginfo("Received 0 — no action.")
        else:
            rospy.logwarn(f"Received invalid command: {command}")

    def shutdown(self):
        moveit_commander.roscpp_shutdown()
        rospy.loginfo("MoveIt Commander shut down.")


if __name__ == "__main__":
    try:
        controller = UR10eMoveItController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
