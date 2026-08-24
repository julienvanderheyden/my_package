#!/usr/bin/env python3
import rospy
import sys
import math
import numpy as np
from my_package.srv import (
    GetGraspPose, GetGraspPoseRequest,
)
from pcl_package.srv import GetStableEstimate, GetStableEstimateRequest

import moveit_commander

# Outer/inner retry limits (see plan_grasp / plan_and_confirm docstrings)
MAX_PLANNING_ATTEMPTS = 3   # replans of the SAME target pose before giving up on it
MAX_GRASP_CYCLES = 3        # full re-perceive + re-plan-grasp cycles before giving up entirely

# plan_and_confirm outcomes - distinguishes "planning itself failed" (worth
# retrying) from "planning succeeded but the user chose not to execute"
# (not a failure - should NOT trigger a retry).
PLAN_SUCCESS = "success"
PLAN_FAILED = "planning_failed"
PLAN_DECLINED = "user_declined"


def is_crazy_plan(plan):
    n_points = len(plan.joint_trajectory.points)
    if n_points <= 0:
        return True
    traj = np.array([p.positions for p in plan.joint_trajectory.points])
    joint_sweep = [round(math.degrees(v), 1) for v in (traj.max(axis=0) - traj.min(axis=0))]
    return any(sweep > 180.0 for sweep in joint_sweep)

def plan_and_confirm(mgc, target_pose, stage_name, max_attempts=MAX_PLANNING_ATTEMPTS):
    """Plans towards target_pose (a geometry_msgs/Pose), retrying up to
    max_attempts times if planning fails or produces a crazy trajectory -
    motion planners like RRTConnect are stochastic, so a fresh attempt from
    the same start/goal can succeed even if the previous one didn't.
    Returns PLAN_SUCCESS (plan executed), PLAN_DECLINED (a valid plan was
    found but the user chose not to run it - not a failure, not retried
    here), or PLAN_FAILED (no valid plan after max_attempts)."""
    for attempt in range(1, max_attempts + 1):
        rospy.loginfo(f"--- Planning for {stage_name} (attempt {attempt}/{max_attempts}) ---")
        mgc.set_start_state_to_current_state()
        mgc.set_pose_target(target_pose)

        success, plan, planning_time, error_code = mgc.plan()
        mgc.clear_pose_targets()

        if success and not is_crazy_plan(plan):
            rospy.loginfo(f"Plan for {stage_name} successfully computed "
                           f"(attempt {attempt}/{max_attempts}).")
            user_input = input(f"Do you want to execute the {stage_name} plan? [y/N]: ").strip().lower()
            if user_input in ['y', 'yes']:
                rospy.loginfo(f"Executing {stage_name}...")
                mgc.execute(plan, wait=True)
                rospy.loginfo(f"{stage_name} execution complete.")
                return PLAN_SUCCESS
            else:
                rospy.loginfo(f"Execution of {stage_name} aborted by user.")
                return PLAN_DECLINED

        rospy.logwarn(f"Planning failed or produced an invalid trajectory for {stage_name} "
                       f"(attempt {attempt}/{max_attempts}).")

    rospy.logerr(f"All {max_attempts} planning attempts failed for {stage_name}.")
    return PLAN_FAILED

def add_collision_object(scene, est, object_name="target_object"):
    """Adds the perceived primitive estimate into the MoveIt planning scene."""
    p_type = est.primitive_type
    pose = est.pose

    inflation_factor = 1.5  # Inflate dimensions to provide a safety margin
    
    if p_type == "CYLINDER":
        # height, radius
        scene.add_cylinder(object_name, pose, est.height*inflation_factor, est.diameter*inflation_factor / 2.0)
    elif p_type == "SPHERE":
        # radius
        scene.add_sphere(object_name, pose, est.diameter*inflation_factor / 2.0)
    elif p_type in ["FLAT_BOX", "BOX"]:
        # size tuple (x, y, z) -> (width, thickness, depth)
        scene.add_box(object_name, pose, (est.width*inflation_factor, est.depth*inflation_factor, est.thickness*inflation_factor))
    else:
        rospy.logwarn(f"Unknown primitive_type '{p_type}'. Skipping collision object creation.")

def get_grasp_candidate(get_estimate, get_grasp):
    """Runs one full perception + grasp-planning cycle. Returns (est,
    grasp_response) on success, or (None, None) if either step failed - the
    caller decides whether/how to retry."""
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

def attempt_grasp_cycle(mgc, scene, get_estimate, get_grasp, obj_name, cycle_num):
    """One full attempt: get a grasp candidate, plan+execute the approach,
    then plan+execute the final grasp. Returns:
      PLAN_SUCCESS  - full sequence completed
      PLAN_DECLINED - user chose not to execute a valid plan (stop, don't retry)
      PLAN_FAILED   - planning failed even after MAX_PLANNING_ATTEMPTS retries
                      (caller may re-query a fresh grasp candidate and retry)
      None          - perception or grasp planning itself failed (caller may retry)
    """
    rospy.loginfo(f"=== Grasp cycle {cycle_num}/{MAX_GRASP_CYCLES}: requesting a grasp candidate ===")
    est, grasp_response = get_grasp_candidate(get_estimate, get_grasp)
    if est is None:
        return None

    rospy.loginfo(f"Adding collision object '{obj_name}' to the planning scene.")
    add_collision_object(scene, est.estimate, object_name=obj_name)
    rospy.sleep(0.5)  # Small delay to ensure the scene updates over ROS topics

    try:
        # Set higher speed limits for approach :
        # - 0.8 : veryyy fast
        # - 0.5 : fast but reasonable
        # - 0.3 : medium
        # - 0.1 : slow
        mgc.set_max_velocity_scaling_factor(0.5)
        mgc.set_max_acceleration_scaling_factor(0.5)

        approach_outcome = plan_and_confirm(mgc, grasp_response.approach_pose_flange, "APPROACH")
        if approach_outcome != PLAN_SUCCESS:
            return approach_outcome

        # Remove collision object prior to the final grasp phase
        rospy.loginfo(f"Removing collision object '{obj_name}' for final grasp execution.")
        scene.remove_world_object(obj_name)
        rospy.sleep(0.5)

        # Set lower speed limits for the final grasp
        mgc.set_max_velocity_scaling_factor(0.1)
        mgc.set_max_acceleration_scaling_factor(0.1)

        grasp_outcome = plan_and_confirm(mgc, grasp_response.grasp_pose_flange, "FINAL GRASP")
        if grasp_outcome == PLAN_SUCCESS:
            rospy.loginfo("Full grasp sequence completed successfully!")
        return grasp_outcome

    finally:
        # Clean up scene object regardless of outcome (no-op if already removed)
        scene.remove_world_object(obj_name)

def plan_grasp():
    rospy.init_node('reaching_node')

    rospy.wait_for_service('/perception/get_stable_estimate')
    get_estimate = rospy.ServiceProxy('/perception/get_stable_estimate', GetStableEstimate)

    rospy.wait_for_service("/grasp_planning/get_grasp_pose")
    get_grasp = rospy.ServiceProxy("/grasp_planning/get_grasp_pose", GetGraspPose)

    moveit_commander.roscpp_initialize(sys.argv)
    mgc = moveit_commander.MoveGroupCommander("right_arm")
    scene = moveit_commander.PlanningSceneInterface()

    mgc.set_planner_id("RRTConnectkConfigDefault")
    mgc.set_num_planning_attempts(5)

    rospy.loginfo("Arm Controller initialized")
    rospy.loginfo(f"Planning frame: {mgc.get_planning_frame()}")
    rospy.loginfo(f"End effector link: {mgc.get_end_effector_link()}")

    obj_name = "target_object"

    for cycle in range(1, MAX_GRASP_CYCLES + 1):
        outcome = attempt_grasp_cycle(mgc, scene, get_estimate, get_grasp, obj_name, cycle)

        if outcome == PLAN_SUCCESS:
            return
        if outcome == PLAN_DECLINED:
            rospy.loginfo("Stopping: user declined to execute a valid plan.")
            return
        if outcome is None:
            rospy.logwarn(f"Perception/grasp planning failed on cycle {cycle}/{MAX_GRASP_CYCLES}.")
        else:  # PLAN_FAILED
            rospy.logwarn(f"Motion planning failed on cycle {cycle}/{MAX_GRASP_CYCLES} - "
                           f"requesting a fresh grasp candidate.")

        if cycle < MAX_GRASP_CYCLES:
            rospy.sleep(0.5)

    rospy.logerr(f"Giving up after {MAX_GRASP_CYCLES} full grasp cycles.")


if __name__ == '__main__':
    plan_grasp()