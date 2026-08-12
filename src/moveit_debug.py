#!/usr/bin/env python3
import sys, math, time
import rospy
import moveit_commander
from itertools import chain
from copy import deepcopy
from tf.transformations import quaternion_multiply, quaternion_from_euler
from geometry_msgs.msg import Quaternion
import numpy as np
from geometry_msgs.msg import Quaternion, PoseStamped  # Added PoseStamped
from visualization_msgs.msg import Marker              # Added Marker

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('handeye_pose_diagnostic', anonymous=True)
mgc = moveit_commander.MoveGroupCommander("right_arm")
mgc.set_planner_id("RRTConnectkConfigDefault")  # same as easy_handeye - test if this is even valid

# --- VISUALIZATION PUBLISHERS ---
pose_pub = rospy.Publisher('/target_pose', PoseStamped, queue_size=1, latch=True)
marker_pub = rospy.Publisher('/target_marker', Marker, queue_size=1, latch=True)

planning_frame = mgc.get_planning_frame()
eef_link = mgc.get_end_effector_link()
rospy.loginfo(f"Arm Controller initialized")
rospy.loginfo(f"Planning frame: {planning_frame}")
rospy.loginfo(f"End effector link: {eef_link}")

start_pose = mgc.get_current_pose()
target_pose = deepcopy(start_pose)
target_pose.pose.position.x += 0.3
target_pose.pose.position.y -= 0.0
target_pose.pose.position.z -= 0.4

target_pose.pose.orientation = Quaternion(*quaternion_multiply(
    [target_pose.pose.orientation.x, target_pose.pose.orientation.y, target_pose.pose.orientation.z, target_pose.pose.orientation.w],
    quaternion_from_euler(-math.pi/2, 0.0, 0.0)
))


# target_pose.pose.position.x = 1.34
# target_pose.pose.position.y = 0.25
# target_pose.pose.position.z = 0.85

# target_pose.pose.orientation.x = -0.0767
# target_pose.pose.orientation.y = 0.7101
# target_pose.pose.orientation.z = -0.0872
# target_pose.pose.orientation.w = -0.6944

# --- PUBLISH TARGET TO RVIZ ---
pose_pub.publish(target_pose)

marker = Marker()
marker.header = target_pose.header
marker.type = Marker.SPHERE
marker.action = Marker.ADD
marker.pose = target_pose.pose
marker.scale.x = 0.05
marker.scale.y = 0.05
marker.scale.z = 0.05
marker.color.a = 1.0
marker.color.r = 1.0  # Red sphere at target position
marker_pub.publish(marker)



# --- PLAN TO TARGET ---
mgc.set_pose_target(target_pose)
t0 = time.time()
success, plan, planning_time, error_code = mgc.plan()
elapsed = time.time() - t0
n_points = len(plan.joint_trajectory.points)
if n_points > 0:
    traj = np.array([p.positions for p in plan.joint_trajectory.points])
    joint_sweep = [round(math.degrees(v), 1) for v in (traj.max(axis=0) - traj.min(axis=0))]
else:
    joint_sweep = None
print(f"Target pose: success={success} error={error_code} time={elapsed:.2f}s points={n_points} sweep_deg={joint_sweep}")

rospy.sleep(5.0)  # Short sleep so latched topics remain available in RViz