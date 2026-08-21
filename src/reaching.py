#!/usr/bin/env python3
import rospy
import sys
import math
import numpy as np
from my_package.srv import (
    GetCylinderGraspPose, GetCylinderGraspPoseRequest,
    GetFlatBoxGraspPose, GetFlatBoxGraspPoseRequest,
    GetSphereGraspPose, GetSphereGraspPoseRequest,
)
from pcl_package.srv import GetStableEstimate, GetStableEstimateRequest

import moveit_commander


# Maps a perceived primitive_type string to the grasp-planning service that
# handles it: (service_topic, request_type). Add new primitives here as new
# grasp planner services come online.
GRASP_SERVICES = {
    "CYLINDER": ("/grasp_planning/get_cylinder_grasp", GetCylinderGraspPose, GetCylinderGraspPoseRequest),
    "FLAT_BOX": ("/grasp_planning/get_flatbox_grasp", GetFlatBoxGraspPose, GetFlatBoxGraspPoseRequest),
    "SPHERE": ("/grasp_planning/get_sphere_grasp", GetSphereGraspPose, GetSphereGraspPoseRequest),
}


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
    else:
        mgc.set_pose_target(target_pose)

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

def get_grasp_for_estimate(est):
    """Dispatches to the grasp-planning service registered for
    est.primitive_type in GRASP_SERVICES, and returns its response (or None
    on failure/unknown type)."""
    primitive_type = est.primitive_type
    entry = GRASP_SERVICES.get(primitive_type)
    if entry is None:
        rospy.logwarn(f"No grasp planner registered for primitive_type '{primitive_type}' "
                       f"(known types: {list(GRASP_SERVICES.keys())})")
        return None

    service_topic, service_type, request_type = entry

    rospy.wait_for_service(service_topic)
    get_grasp = rospy.ServiceProxy(service_topic, service_type)

    grasp_req = request_type()
    grasp_req.estimate = est
    return get_grasp, grasp_req

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

def plan_grasp():
    rospy.init_node('reaching_node')

    rospy.wait_for_service('/perception/get_stable_estimate')
    get_estimate = rospy.ServiceProxy('/perception/get_stable_estimate', GetStableEstimate)

    grasp_response = None
    target_est = None
    try:
        est = get_estimate()
        if not est.success:
            rospy.logerr(f"Perception failed. Reason: {est.reason}")
            return

        target_est = est.estimate
        dispatch = get_grasp_for_estimate(target_est)
        if dispatch is None:
            return
        get_grasp, grasp_req = dispatch
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
    scene = moveit_commander.PlanningSceneInterface()
    
    mgc.set_planner_id("RRTConnectkConfigDefault")
    mgc.set_num_planning_attempts(5)

    rospy.loginfo("Arm Controller initialized")
    rospy.loginfo(f"Planning frame: {mgc.get_planning_frame()}")
    rospy.loginfo(f"End effector link: {mgc.get_end_effector_link()}")

    obj_name = "target_object"

    try:
        # Step 1: Add collision object for the approach phase
        rospy.loginfo(f"Adding collision object '{obj_name}' to the planning scene.")
        add_collision_object(scene, target_est, object_name=obj_name)
        rospy.sleep(0.5)  # Small delay to ensure the scene updates over ROS topics

        # Set higher speed limits for approach : 
        # - 0.8 : veryyy fast
        # - 0.5 : fast but reasonable
        # - 0.3 : medium
        # - 0.1 : slow
        mgc.set_max_velocity_scaling_factor(0.5)
        mgc.set_max_acceleration_scaling_factor(0.5)

        approach_executed = plan_and_confirm(mgc, grasp_response.approach_pose_flange, "APPROACH")
        if not approach_executed:
            return

        # Step 2: Remove collision object prior to the final grasp phase
        rospy.loginfo(f"Removing collision object '{obj_name}' for final grasp execution.")
        scene.remove_world_object(obj_name)
        rospy.sleep(0.5)

        # Set lower speed limits for the final grasp
        mgc.set_max_velocity_scaling_factor(0.1)
        mgc.set_max_acceleration_scaling_factor(0.1)

        grasp_executed = plan_and_confirm(mgc, grasp_response.grasp_pose_flange, "FINAL GRASP")
        if grasp_executed:
            rospy.loginfo("Full grasp sequence completed successfully!")

    finally:
        # Clean up scene object if execution is aborted or crashes midway
        scene.remove_world_object(obj_name)


if __name__ == '__main__':
    plan_grasp()