#!/usr/bin/env python3
import rospy
from my_package.srv import (
    GetGraspPose, GetGraspPoseRequest,
    MoveToPose, MoveToPoseRequest
)
from pcl_package.srv import GetStableEstimate, GetStableEstimateRequest

MAX_GRASP_CYCLES = 3

PLAN_SUCCESS = "success"
PLAN_FAILED = "planning_failed"
PLAN_DECLINED = "user_declined"


def call_move_to_pose(move_srv, target_pose, estimate=None, wait_for_confirmation=True, velocity_scaling=0.5):
    """Wraps calls to the /arm_motion/move_to_pose service."""
    req = MoveToPoseRequest()
    req.target_pose = target_pose
    if estimate is not None:
        req.estimate = estimate
    req.wait_for_confirmation = wait_for_confirmation
    req.velocity_scaling = velocity_scaling

    try:
        res = move_srv(req)
        return res.outcome
    except rospy.ServiceException as e:
        rospy.logerr(f"MoveToPose service call failed: {e}")
        return PLAN_FAILED


def get_grasp_candidate(get_estimate, get_grasp):
    """Runs one full perception + grasp-planning cycle."""
    try:
        est = get_estimate()
    except rospy.ServiceException as e:
        rospy.logerr(f"Perception service call failed: {e}")
        return None, None

    if not est.success:
        rospy.logerr(f"Perception failed. Reason: {est.reason}")
        return None, None

    try:
        grasp_req = GetGraspPoseRequest()
        grasp_req.estimate = est.estimate
        grasp_req.cloud = est.cloud
        grasp_response = get_grasp(grasp_req)
    except rospy.ServiceException as e:
        rospy.logerr(f"Grasp planning service call failed: {e}")
        return None, None

    if not grasp_response.success:
        rospy.logerr(f"Grasp planning failed. Reason: {grasp_response.reason}")
        return None, None

    return est, grasp_response


def attempt_grasp_cycle(get_estimate, get_grasp, move_srv, cycle_num):
    """One full attempt: get a grasp candidate, delegate approach and final grasp moves."""
    rospy.loginfo(f"=== Grasp cycle {cycle_num}/{MAX_GRASP_CYCLES}: requesting a grasp candidate ===")
    est, grasp_response = get_grasp_candidate(get_estimate, get_grasp)
    if est is None:
        return None

    # Step 1: APPROACH (Include collision object, fast velocity scaling)
    rospy.loginfo("Executing APPROACH phase...")
    approach_outcome = call_move_to_pose(
        move_srv,
        target_pose=grasp_response.approach_pose_flange,
        estimate=est.estimate,  # Serves to add collision object during approach
        wait_for_confirmation=True,
        velocity_scaling=0.5
    )
    if approach_outcome != PLAN_SUCCESS:
        return approach_outcome

    # Step 2: FINAL GRASP (Omit collision object to allow contact, slow velocity scaling)
    rospy.loginfo("Executing FINAL GRASP phase...")
    grasp_outcome = call_move_to_pose(
        move_srv,
        target_pose=grasp_response.grasp_pose_flange,
        estimate=None,  # Omit estimate to allow contact with object
        wait_for_confirmation=True,
        velocity_scaling=0.1
    )
    if grasp_outcome == PLAN_SUCCESS:
        rospy.loginfo("Full grasp sequence completed successfully!")
    return grasp_outcome


def plan_grasp():
    rospy.init_node('reaching_orchestrator')

    rospy.wait_for_service('/perception/get_stable_estimate')
    get_estimate = rospy.ServiceProxy('/perception/get_stable_estimate', GetStableEstimate)

    rospy.wait_for_service("/grasp_planning/get_grasp_pose")
    get_grasp = rospy.ServiceProxy("/grasp_planning/get_grasp_pose", GetGraspPose)

    rospy.wait_for_service("/arm_motion/move_to_pose")
    move_srv = rospy.ServiceProxy("/arm_motion/move_to_pose", MoveToPose)

    rospy.loginfo("Orchestrator node initialized and services connected.")

    for cycle in range(1, MAX_GRASP_CYCLES + 1):
        outcome = attempt_grasp_cycle(get_estimate, get_grasp, move_srv, cycle)

        if outcome == PLAN_SUCCESS:
            return
        if outcome == PLAN_DECLINED:
            rospy.loginfo("Stopping: user declined to execute a valid plan.")
            return
        if outcome is None:
            rospy.logwarn(f"Perception/grasp planning failed on cycle {cycle}/{MAX_GRASP_CYCLES}.")
        else:  # PLAN_FAILED
            rospy.logwarn(f"Motion planning failed on cycle {cycle}/{MAX_GRASP_CYCLES} - requesting fresh candidate.")

        if cycle < MAX_GRASP_CYCLES:
            rospy.sleep(0.5)

    rospy.logerr(f"Giving up after {MAX_GRASP_CYCLES} full grasp cycles.")


if __name__ == '__main__':
    plan_grasp()