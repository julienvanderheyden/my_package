#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
import geometry_msgs.msg
from math import pi
from tf.transformations import quaternion_from_euler

def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('ur10e_cartesian_goal', anonymous=True)

    group_name = "right_arm"
    move_group = moveit_commander.MoveGroupCommander(group_name)

    planning_frame = move_group.get_planning_frame()
    eef_link = move_group.get_end_effector_link()
    rospy.loginfo(f"Planning frame: {planning_frame}")
    rospy.loginfo(f"End effector link: {eef_link}")

    # Define a reachable pose (modify if needed)
    target_pose = geometry_msgs.msg.Pose()
    target_pose.position.x = 0.3
    target_pose.position.y = 0.0
    target_pose.position.z = 0.5

    roll, pitch, yaw = 0, pi/2, 0
    q = quaternion_from_euler(roll, pitch, yaw)
    target_pose.orientation.x = q[0]
    target_pose.orientation.y = q[1]
    target_pose.orientation.z = q[2]
    target_pose.orientation.w = q[3]

    move_group.set_pose_target(target_pose, eef_link)

    rospy.loginfo("Planning trajectory to target pose...")
    plan_result = move_group.plan()

    # Handle MoveIt versions returning tuples
    if isinstance(plan_result, tuple):
        plan = plan_result[1] if len(plan_result) > 1 else plan_result[0]
    else:
        plan = plan_result

    if not hasattr(plan, 'joint_trajectory') or len(plan.joint_trajectory.points) == 0:
        rospy.logerr("Planning failed!")
        return

    rospy.loginfo("Executing trajectory...")
    #move_group.execute(plan, wait=True)

    move_group.stop()
    move_group.clear_pose_targets()
    rospy.loginfo("Motion complete.")

    moveit_commander.roscpp_shutdown()

if __name__ == "__main__":
    main()
