#!/usr/bin/env python3
"""
arm_motion_node.py

Service node: moves the UR10e arm to a single target pose, wrapping the
same plan/execute logic reaching.py used inline (replanning on failure,
rejecting "crazy" joint-swing plans, optional human confirmation before
execution) behind a stable service interface so a higher-level orchestrator
doesn't need to know anything about MoveIt.

One service call = one move. Collision-object avoidance is scoped to that
single call: if the request carries a PrimitiveEstimate (non-empty
primitive_type), it's added to the planning scene before planning this move
and removed again immediately afterwards - success, failure, or decline -
never left lingering for a later call. This lets the SAME service serve
both the "avoid the object en route" case (APPROACH) and the "allow
contact" case (FINAL GRASP, DROP, HOME) - the caller just includes or omits
the estimate. Collision checking is independent of goal representation, so
this applies whether the request carries a Cartesian pose or a joint-space
target (see below) - MoveIt still avoids scene objects when planning to a
joint goal.

Two target modes, selected by req.use_joint_target:
  - False (default): plan to req.target_pose (Cartesian, via set_pose_target).
  - True: plan to the explicit joint configuration given by the six ra_*
    fields on the request (via set_joint_value_target), ignoring
    target_pose entirely. Used for HOME, where a single known, repeatable
    joint configuration is wanted rather than whichever IK solution MoveIt
    happens to pick for a Cartesian goal.

Outer retries (e.g. "if planning keeps failing, ask the grasp planner for
a different candidate pose") are deliberately NOT handled here - that's
sequencing logic that belongs to the orchestrator, not the arm. This node
only replans the SAME target a bounded number of times before giving up
and reporting PLAN_FAILED.
"""

import sys
import math

import rospy
import numpy as np

import moveit_commander

from my_package.srv import MoveToPose, MoveToPoseResponse


OBJECT_NAME = "target_object"
MAX_REPLAN_ATTEMPTS_DEFAULT = 3
DEFAULT_VELOCITY_SCALING = 0.5

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


class ArmMotionNode(object):
    def __init__(self):
        self.planning_group = rospy.get_param("~planning_group", "right_arm")
        self.planner_id = rospy.get_param("~planner_id", "RRTConnectkConfigDefault")
        self.num_planning_attempts = rospy.get_param("~num_planning_attempts", 5)
        self.max_replan_attempts = rospy.get_param("~max_replan_attempts", MAX_REPLAN_ATTEMPTS_DEFAULT)

        moveit_commander.roscpp_initialize(sys.argv)
        self.mgc = moveit_commander.MoveGroupCommander(self.planning_group)
        self.scene = moveit_commander.PlanningSceneInterface()

        self.mgc.set_planner_id(self.planner_id)
        self.mgc.set_num_planning_attempts(self.num_planning_attempts)

        rospy.loginfo("Arm motion node initialized (group='%s', planner='%s')",
                       self.planning_group, self.planner_id)
        rospy.loginfo("Planning frame: %s", self.mgc.get_planning_frame())
        rospy.loginfo("End effector link: %s", self.mgc.get_end_effector_link())

        self.move_service = rospy.Service("/arm_motion/move_to_pose", MoveToPose, self.handle_move_to_pose)
        rospy.loginfo("Arm motion service ready on /arm_motion/move_to_pose")

    # -- Collision object management (scoped to a single service call) -----
    def _add_collision_object(self, estimate):
        """Adds estimate as a MoveIt collision object. Returns True if
        something was actually added (so the caller knows whether removal
        is needed afterwards)."""
        p_type = estimate.primitive_type
        pose = estimate.pose
        inflation_factor = 1.5  # safety margin

        if p_type == "CYLINDER":
            self.scene.add_cylinder(OBJECT_NAME, pose, estimate.height * inflation_factor,
                                     estimate.diameter * inflation_factor / 2.0)
        elif p_type == "SPHERE":
            self.scene.add_sphere(OBJECT_NAME, pose, estimate.diameter * inflation_factor / 2.0)
        elif p_type in ("FLAT_BOX", "BOX"):
            self.scene.add_box(OBJECT_NAME, pose,
                                (estimate.width * inflation_factor,
                                 estimate.depth * inflation_factor,
                                 estimate.thickness * inflation_factor))
        else:
            rospy.logwarn("Unknown primitive_type '%s' - skipping collision object creation.", p_type)
            return False

        rospy.sleep(0.5)  # let the scene update propagate over ROS topics
        return True

    def _remove_collision_object(self):
        self.scene.remove_world_object(OBJECT_NAME)

    # -- Plan/execute, with replanning + optional confirmation --------------
    def _plan_and_execute(self, req, wait_for_confirmation, max_attempts):
        """replans up to max_attempts times (motion planners like RRTConnect are
        stochastic, so a fresh attempt can succeed where the last one didn't), 
        then either executes immediately or blocks for a y/N confirmation depending 
        on wait_for_confirmation. Returns (outcome, reason).

        req.use_joint_target selects the goal representation: a Cartesian
        target_pose (set_pose_target) or an explicit joint configuration
        (set_joint_value_target) built from the six named ra_* fields on
        the request. NEVER pass a Pose to set_joint_value_target or a
        joint dict to set_pose_target - MoveIt won't raise, it'll just
        silently produce a nonsensical plan."""
        if req.use_joint_target:
            joint_goal = {
                "ra_shoulder_pan_joint": req.ra_shoulder_pan_joint,
                "ra_shoulder_lift_joint": req.ra_shoulder_lift_joint,
                "ra_elbow_joint": req.ra_elbow_joint,
                "ra_wrist_1_joint": req.ra_wrist_1_joint,
                "ra_wrist_2_joint": req.ra_wrist_2_joint,
                "ra_wrist_3_joint": req.ra_wrist_3_joint,
            }
            rospy.loginfo("Target: joint configuration %s", joint_goal)
        else:
            rospy.loginfo("Target: Cartesian pose %s", req.target_pose)

        for attempt in range(1, max_attempts + 1):
            rospy.loginfo("--- Planning move (attempt %d/%d) ---", attempt, max_attempts)
            self.mgc.set_start_state_to_current_state()

            if req.use_joint_target:
                self.mgc.set_joint_value_target(joint_goal)
            else:
                self.mgc.set_pose_target(req.target_pose)

            success, plan, planning_time, error_code = self.mgc.plan()
            self.mgc.clear_pose_targets()  # no-op when a joint target was used (nothing to clear);
                                            # kept unconditional so cleanup doesn't depend on mode

            if success and not is_crazy_plan(plan):
                rospy.loginfo("Valid plan found (attempt %d/%d).", attempt, max_attempts)

                if wait_for_confirmation:
                    user_input = input("Execute this plan? [y/N]: ").strip().lower()
                    if user_input not in ('y', 'yes'):
                        rospy.loginfo("Execution declined by user.")
                        return PLAN_DECLINED, "User declined to execute the planned motion."

                rospy.loginfo("Executing move...")
                self.mgc.execute(plan, wait=True)
                rospy.loginfo("Move execution complete.")
                return PLAN_SUCCESS, ""

            rospy.logwarn(f"Planning failed or produced an invalid trajectory (attempt {attempt}/{max_attempts}). Error code: {error_code}")

        reason = f"No valid plan found after {max_attempts} attempts."
        rospy.logerr(reason)
        return PLAN_FAILED, reason

    # -- Service handler -----------------------------------------------------
    def handle_move_to_pose(self, req):
        res = MoveToPoseResponse()

        has_estimate = bool(req.estimate.primitive_type)
        added_collision_object = False
        if has_estimate:
            rospy.loginfo("Adding collision object for primitive_type '%s'.", req.estimate.primitive_type)
            added_collision_object = self._add_collision_object(req.estimate)

        velocity_scaling = req.velocity_scaling if req.velocity_scaling > 0.0 else DEFAULT_VELOCITY_SCALING
        velocity_scaling = min(velocity_scaling, 1.0)
        self.mgc.set_max_velocity_scaling_factor(velocity_scaling)
        self.mgc.set_max_acceleration_scaling_factor(velocity_scaling)

        try:
            outcome, reason = self._plan_and_execute(
                req, req.wait_for_confirmation, self.max_replan_attempts)
        finally:
            # Always clean up, regardless of success/failure/decline, so a
            # stale collision object never leaks into the next call.
            if added_collision_object:
                self._remove_collision_object()

        res.success = (outcome == PLAN_SUCCESS)
        res.outcome = outcome
        res.reason = reason
        return res


def main():
    rospy.init_node("arm_motion_node")
    ArmMotionNode()
    rospy.spin()


if __name__ == "__main__":
    main()