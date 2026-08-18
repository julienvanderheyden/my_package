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

def plan_and_confirm(mgc, target_pose, stage_name):
    """Helper to set start state, set target, plan, check sanity, and prompt user."""
    rospy.loginfo(f"--- Planning for {stage_name} ---")
    mgc.set_start_state_to_current_state()
    
    # Determine target type (Joint values or Pose)
    if isinstance(target_pose, (list, tuple)):
        mgc.set_joint_value_target(target_pose)
        print(f"Planning to joint target")
    else:
        mgc.set_pose_target(target_pose)
        print(f"Planning to pose target")

    success, plan, planning_time, error_code = mgc.plan()

    if not (success and not is_crazy_plan(plan)):
        rospy.logwarn(f"Planning failed or produced an invalid trajectory for {stage_name}.")
        return False

    rospy.loginfo(f"Plan for {stage_name} successfully computed!")
    
    user_input = input(f"Do you want to execute the {stage_name} plan? [y/N]: ").strip().lower()
    if user_input in ['y', 'yes']:
        rospy.loginfo(f"Executing {stage_name}...")
        mgc.execute(plan, wait=True)
        rospy.loginfo(f"{stage_name} execution complete.")
        return True
    else:
        rospy.loginfo(f"Execution of {stage_name} aborted by user.")
        return False

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
            rospy.logwarn(f"Expected CYLINDER, but got '{est.estimate.primitive_type}'")
            return
        else:
            grasp_req = GetCylinderGraspPoseRequest()
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
    mgc.set_num_planning_attempts(5)

    rospy.loginfo("Arm Controller initialized")
    rospy.loginfo(f"Planning frame: {mgc.get_planning_frame()}")
    rospy.loginfo(f"End effector link: {mgc.get_end_effector_link()}")

    # Step 1: Approach Phase
    approach_executed = plan_and_confirm(mgc, grasp_response.approach_pose_flange, "APPROACH")
    if not approach_executed:
        return

    # Step 2: Final Grasp Phase (Planned only after approach executes successfully)
    grasp_executed = plan_and_confirm(mgc, grasp_response.grasp_pose_flange, "FINAL GRASP")
    if grasp_executed:
        rospy.loginfo("Full grasp sequence completed successfully!")


if __name__ == '__main__':
    plan_grasp()