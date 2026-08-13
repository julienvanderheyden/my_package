#!/usr/bin/env python3
import rospy
import sys
import math
import numpy as np
from my_package.srv import GetCylinderGraspPose, GetCylinderGraspPoseRequest
from pcl_package.srv import GetStableEstimate, GetStableEstimateRequest

import moveit_commander


def is_crazy_plan(plan):
    n_points = len(plan.joint_trajectory.points)
    if n_points <= 0:
        return True
    traj = np.array([p.positions for p in plan.joint_trajectory.points])
    joint_sweep = [round(math.degrees(v), 1) for v in (traj.max(axis=0) - traj.min(axis=0))]
    return any(sweep > 180.0 for sweep in joint_sweep)

def plan_grasp():
    rospy.init_node('reaching_node')

    # Wait for services and setup proxies outside the try block
    rospy.wait_for_service('/perception/get_stable_estimate')
    rospy.wait_for_service('/grasp_planning/get_cylinder_grasp')

    get_estimate = rospy.ServiceProxy('/perception/get_stable_estimate', GetStableEstimate)
    get_grasp = rospy.ServiceProxy('/grasp_planning/get_cylinder_grasp', GetCylinderGraspPose)

    grasp_response = None
    try:
        est = get_estimate()
        if not est.success:
            rospy.logerr(f"Perception failed. Reason: {est.reason}")
            return
        
        elif est.estimate.primitive_type != "CYLINDER":
            rospy.logwarn(f"Expected CYLINDER, but got '{est.primitive_type}'")
            return
        else:
            grasp_req = GetCylinderGraspPoseRequest()
            grasp_req.primitive_type = est.estimate.primitive_type
            grasp_req.estimate = est.estimate
            grasp_req.cloud = est.cloud

            grasp_response = get_grasp(grasp_req)

    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

    if not grasp_response or not grasp_response.success:
        reason = grasp_response.reason if grasp_response else "No response received"
        rospy.logerr(f"Grasp planning failed. Reason: {reason}")
        return 

    moveit_commander.roscpp_initialize(sys.argv)
    mgc = moveit_commander.MoveGroupCommander("right_arm")
    mgc.set_planner_id("RRTConnectkConfigDefault")

    planning_frame = mgc.get_planning_frame()
    eef_link = mgc.get_end_effector_link()
    rospy.loginfo(f"Arm Controller initialized")
    rospy.loginfo(f"Planning frame: {planning_frame}")
    rospy.loginfo(f"End effector link: {eef_link}")

    mgc.set_start_state_to_current_state()
    mgc.set_joint_value_target(grasp_response.approach_pose_flange)
    success, plan, planning_time, error_code = mgc.plan()

    if success and not is_crazy_plan(plan):
        rospy.loginfo("Grasp plan successfully computed!")
        
        # Ask user for confirmation in the terminal
        user_input = input("Do you want to execute this grasp plan? [y/N]: ").strip().lower()
        if user_input in ['y', 'yes']:
            rospy.loginfo("Executing plan...")
            mgc.execute(plan, wait=True)
            rospy.loginfo("Execution complete.")
        else:
            rospy.loginfo("Execution aborted by user.")
    else:
        rospy.logwarn("Planning failed or produced an invalid trajectory.")


if __name__ == '__main__':
    plan_grasp()