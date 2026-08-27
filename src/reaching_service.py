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
this applies whether the request carries a Cartesian pose, a joint-space
target, or a Cartesian-path target (see below) - MoveIt still avoids scene
objects in every case.

Three motion modes, selected by req.motion_mode:
  - MOTION_POSE (default): plan to req.target_pose with the sampling-based
    planner (set_pose_target + RRTConnect).
  - MOTION_JOINT: plan to the explicit joint configuration given by the six
    named ra_* fields on the request (set_joint_value_target), ignoring
    target_pose entirely. Used for HOME, where a single known, repeatable
    joint configuration is wanted rather than whichever IK solution MoveIt
    happens to pick for a Cartesian goal.
  - MOTION_CARTESIAN: plan a straight-line interpolated path to
    req.target_pose (compute_cartesian_path) instead of sampling. This is
    what rules out "crazy" large joint-swing plans in the first place -
    there's no random tree to produce one - at the cost of flexibility: a
    straight line that isn't reachable just fails (low fraction_complete)
    rather than finding some other way around. Used for LIFT and DROP,
    where the motion is short, the path shape matters (straight up, then a
    direct line to the drop point), and an object may be held in the hand.
    NOTE: compute_cartesian_path times its output trajectory at full speed
    and ignores set_max_velocity_scaling_factor, so this node re-times it
    manually afterwards (see _rescale_cartesian_trajectory) using the same
    velocity_scaling convention as every other mode (smaller = slower).

Outer retries (e.g. "if planning keeps failing, ask the grasp planner for
a different candidate pose") are deliberately NOT handled here - that's
sequencing logic that belongs to the orchestrator, not the arm. This node
only replans/re-attempts the SAME target a bounded number of times before
giving up and reporting PLAN_FAILED.
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
DEFAULT_CARTESIAN_EEF_STEP = 0.005
DEFAULT_CARTESIAN_MIN_FRACTION = 0.95
MIN_VELOCITY_SCALING = 1e-3  # guards the Cartesian re-timing division against a bad/zero request

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
        self.cartesian_eef_step = rospy.get_param("~cartesian_eef_step", DEFAULT_CARTESIAN_EEF_STEP)
        self.cartesian_min_fraction = rospy.get_param("~cartesian_min_fraction", DEFAULT_CARTESIAN_MIN_FRACTION)

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

    # -- Cartesian-path re-timing ---------------------------------------
    def _rescale_cartesian_trajectory(self, plan, velocity_scaling):
        """compute_cartesian_path times its output trajectory at full
        speed and does NOT consult set_max_velocity_scaling_factor, so
        the timing has to be adjusted manually here, after planning.

        velocity_scaling follows the SAME convention as every other
        motion mode in this node: smaller = slower, 1.0 = full speed.
        Concretely, we divide time_from_start by velocity_scaling
        (stretching a slow request out in time) and scale
        velocities/accelerations down to match - so a value like 0.1 (as
        used for the delicate FINAL GRASP move elsewhere in this
        pipeline) means "10% speed", not "10x speed". A naive port of the
        original prototype's `time_from_start *= speed_factor` would have
        inverted this: multiplying time directly by a small scaling
        factor makes the motion FASTER, not slower - exactly backwards
        from how velocity_scaling is used everywhere else in this node,
        and a dangerous mismatch for a motion (LIFT) that's meant to be
        slow because it may be carrying a grasped object."""
        scale = max(min(velocity_scaling, 1.0), MIN_VELOCITY_SCALING)
        inv_scale = 1.0 / scale
        for point in plan.joint_trajectory.points:
            point.time_from_start *= inv_scale
            point.velocities = [v * scale for v in point.velocities]
            point.accelerations = [a * scale * scale for a in point.accelerations]
        return plan

    # -- Plan/execute, with replanning + optional confirmation --------------
    def _plan_attempt(self, req, joint_goal, velocity_scaling):
        """Runs ONE planning attempt for whichever motion mode the request
        selects. Returns (valid, plan, fraction, diagnostic):
          - `fraction` is the achieved Cartesian path fraction for
            MOTION_CARTESIAN, and 1.0 for the sampling-based modes (there's
            no partial-completion concept for those).
          - `diagnostic` is a short human-readable string for the retry log.

        NEVER pass a Pose to set_joint_value_target or a joint dict to
        set_pose_target - MoveIt won't raise, it'll just silently produce
        a nonsensical plan."""
        self.mgc.set_start_state_to_current_state()

        if req.motion_mode == req.MOTION_CARTESIAN:
            eef_step = req.eef_step if req.eef_step > 0.0 else self.cartesian_eef_step
            plan, fraction = self.mgc.compute_cartesian_path(
                [req.target_pose], eef_step=eef_step, jump_threshold=0.0)
            valid = fraction >= self.cartesian_min_fraction and not is_crazy_plan(plan)
            diagnostic = (f"Cartesian path {fraction * 100:.1f}% complete "
                          f"(need >= {self.cartesian_min_fraction * 100:.0f}%)")
            if valid:
                plan = self._rescale_cartesian_trajectory(plan, velocity_scaling)
            return valid, plan, fraction, diagnostic

        if req.motion_mode == req.MOTION_JOINT:
            self.mgc.set_joint_value_target(joint_goal)
        else:
            self.mgc.set_pose_target(req.target_pose)

        success, plan, planning_time, error_code = self.mgc.plan()
        self.mgc.clear_pose_targets()  # no-op for MOTION_JOINT (nothing to clear); kept
                                        # unconditional so cleanup doesn't depend on mode
        valid = success and not is_crazy_plan(plan)
        diagnostic = f"error_code={error_code}"
        return valid, plan, 1.0, diagnostic

    def _plan_and_execute(self, req, wait_for_confirmation, velocity_scaling, max_attempts):
        """Replans/re-attempts up to max_attempts times (RRTConnect is
        stochastic, so a fresh attempt can succeed where the last one
        didn't; a Cartesian path is deterministic given the same start
        state, but retrying is harmless), then either executes immediately
        or blocks for a y/N confirmation depending on wait_for_confirmation.
        Returns (outcome, reason, fraction_complete)."""
        joint_goal = None
        if req.motion_mode == req.MOTION_JOINT:
            joint_goal = {
                "ra_shoulder_pan_joint": req.ra_shoulder_pan_joint,
                "ra_shoulder_lift_joint": req.ra_shoulder_lift_joint,
                "ra_elbow_joint": req.ra_elbow_joint,
                "ra_wrist_1_joint": req.ra_wrist_1_joint,
                "ra_wrist_2_joint": req.ra_wrist_2_joint,
                "ra_wrist_3_joint": req.ra_wrist_3_joint,
            }
            rospy.loginfo("Target: joint configuration %s", joint_goal)
        elif req.motion_mode == req.MOTION_CARTESIAN:
            rospy.loginfo("Target: Cartesian path to pose %s", req.target_pose)
        else:
            rospy.loginfo("Target: Cartesian pose (sampling planner) %s", req.target_pose)

        for attempt in range(1, max_attempts + 1):
            rospy.loginfo("--- Planning move (attempt %d/%d) ---", attempt, max_attempts)
            valid, plan, fraction, diagnostic = self._plan_attempt(req, joint_goal, velocity_scaling)

            if valid:
                rospy.loginfo("Valid plan found (attempt %d/%d): %s", attempt, max_attempts, diagnostic)

                if wait_for_confirmation:
                    user_input = input("Execute this plan? [y/N]: ").strip().lower()
                    if user_input not in ('y', 'yes'):
                        rospy.loginfo("Execution declined by user.")
                        return PLAN_DECLINED, "User declined to execute the planned motion.", fraction

                rospy.loginfo("Executing move...")
                self.mgc.execute(plan, wait=True)
                self.mgc.stop()
                self.mgc.clear_pose_targets()
                rospy.loginfo("Move execution complete.")
                return PLAN_SUCCESS, "", fraction

            rospy.logwarn(f"Planning failed (attempt {attempt}/{max_attempts}): {diagnostic}")

        reason = f"No valid plan/path found after {max_attempts} attempts."
        rospy.logerr(reason)
        return PLAN_FAILED, reason, 0.0

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
            outcome, reason, fraction = self._plan_and_execute(
                req, req.wait_for_confirmation, velocity_scaling, self.max_replan_attempts)
        finally:
            # Always clean up, regardless of success/failure/decline, so a
            # stale collision object never leaks into the next call.
            if added_collision_object:
                self._remove_collision_object()

        res.success = (outcome == PLAN_SUCCESS)
        res.outcome = outcome
        res.reason = reason
        res.fraction_complete = fraction
        return res


def main():
    rospy.init_node("arm_motion_node")
    ArmMotionNode()
    rospy.spin()


if __name__ == "__main__":
    main()