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

angle_delta = math.radians(25)   # matches your rotation_delta_degrees
translation_delta = 0.1          # matches your translation_delta_meters

def compute_poses_around_state(start_pose, angle_delta, translation_delta):
    basis = np.eye(3)
    pos_deltas = [quaternion_from_euler(*rot_axis * angle_delta) for rot_axis in basis]
    neg_deltas = [quaternion_from_euler(*rot_axis * (-angle_delta)) for rot_axis in basis]
    quaternion_deltas = list(chain.from_iterable(zip(pos_deltas, neg_deltas)))
    final_rots = [list(qd) for qd in quaternion_deltas]
    pos_deltas = [quaternion_from_euler(*rot_axis * angle_delta / 2) for rot_axis in basis]
    neg_deltas = [quaternion_from_euler(*rot_axis * (-angle_delta / 2)) for rot_axis in basis]
    quaternion_deltas = list(chain.from_iterable(zip(pos_deltas, neg_deltas)))
    final_rots += [list(qd) for qd in quaternion_deltas]

    final_poses = []
    for rot in final_rots:
        fp = deepcopy(start_pose)
        ori = fp.pose.orientation
        combined = quaternion_multiply([ori.x, ori.y, ori.z, ori.w], rot)
        fp.pose.orientation = Quaternion(*combined)
        final_poses.append(fp)

    for dx in [translation_delta/2, -translation_delta/2]:
        fp = deepcopy(start_pose); fp.pose.position.x += dx; final_poses.append(fp)
    for dy in [translation_delta, -translation_delta]:
        fp = deepcopy(start_pose); fp.pose.position.y += dy; final_poses.append(fp)
    fp = deepcopy(start_pose); fp.pose.position.z += translation_delta/3; final_poses.append(fp)
    return final_poses

start_pose = mgc.get_current_pose()
targets = compute_poses_around_state(start_pose, angle_delta, translation_delta)
print(targets)

for i, t in enumerate(targets):
    mgc.set_pose_target(t)
    t0 = time.time()
    success, plan, planning_time, error_code = mgc.plan()
    elapsed = time.time() - t0
    n_points = len(plan.joint_trajectory.points)
    if n_points > 0:
        traj = np.array([p.positions for p in plan.joint_trajectory.points])
        joint_sweep = [round(math.degrees(v), 1) for v in (traj.max(axis=0) - traj.min(axis=0))]
    else:
        joint_sweep = None
    print(f"Pose {i:2d}: success={success} error={error_code} time={elapsed:.2f}s points={n_points} sweep_deg={joint_sweep}")