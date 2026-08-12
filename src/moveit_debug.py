#!/usr/bin/env python3
import sys, math, time
import rospy
import moveit_commander
from itertools import chain
from copy import deepcopy
from tf.transformations import quaternion_multiply, quaternion_from_euler
from geometry_msgs.msg import Quaternion
import numpy as np

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('handeye_pose_diagnostic', anonymous=True)
mgc = moveit_commander.MoveGroupCommander("right_arm")
mgc.set_planner_id("RRTConnectkConfigDefault")  # same as easy_handeye - test if this is even valid

planning_frame = mgc.get_planning_frame()
eef_link = mgc.get_end_effector_link()
rospy.loginfo(f"Arm Controller initialized")
rospy.loginfo(f"Planning frame: {planning_frame}")
rospy.loginfo(f"End effector link: {eef_link}")

start_pose = mgc.get_current_pose()
target_pose = deepcopy(start_pose)
target_pose.pose.position.x += 0.1


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