#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from tf.transformations import quaternion_from_euler

def main():
    # 1. Initialize MoveIt and ROS node
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('ur10e_cartesian_goal', anonymous=True)

    # 2. Instantiate MoveGroupCommander for the arm
    group_name = "right_arm"   # or "ur10e_arm", depending on your setup
    move_group = moveit_commander.MoveGroupCommander(group_name)

    # Optional: get planning frame and end-effector link
    planning_frame = move_group.get_planning_frame()
    eef_link = move_group.get_end_effector_link()
    rospy.loginfo(f"Planning frame: {planning_frame}")
    rospy.loginfo(f"End effector link: {eef_link}")

    # 3. Define the target pose (position + orientation)
    target_pose = geometry_msgs.msg.Pose()
    target_pose.position.x = 0.40   # meters
    target_pose.position.y = 0.20
    target_pose.position.z = 0.30

    # Orientation as roll-pitch-yaw (in radians)
    roll = 0.0
    pitch = pi / 2
    yaw = 0.0
    q = quaternion_from_euler(roll, pitch, yaw)
    target_pose.orientation.x = q[0]
    target_pose.orientation.y = q[1]
    target_pose.orientation.z = q[2]
    target_pose.orientation.w = q[3]

    move_group.set_pose_target(target_pose)

    # 4. Plan trajectory
    rospy.loginfo("Planning trajectory to target pose...")
    plan = move_group.plan()

    if not plan or len(plan.joint_trajectory.points) == 0:
        rospy.logerr("Planning failed!")
        return

    rospy.loginfo("Planning successful")
    # move_group.go(wait=True)

    # 5. Stop residual motion
    move_group.stop()
    move_group.clear_pose_targets()

    rospy.loginfo("Target reached successfully!")

    # Shutdown
    moveit_commander.roscpp_shutdown()

if __name__ == "__main__":
    main()
