#!/usr/bin/env python3
"""
grasping_orchestrator.py
=========================

High-level orchestrator that runs a continuous
perception -> grasp planning -> preshape -> approach -> grasp ->
closed-loop force grasping -> lift -> drop -> release -> home
pipeline for the Shadow Hand / UR10e platform.

Execution architecture
-----------------------
Outer loop (runs forever, `while not rospy.is_shutdown()`):
    Polls perception for a new target and drives it through the full
    pipeline. After a completed sequence - success OR abandonment - state
    is reset and control returns to perception polling.

Inner retry loop (up to `~max_retries` attempts, default 3, PER TARGET
OBJECT):
    Wraps candidate generation (perception + grasp planning) together with
    the APPROACH and GRASP arm motions. If `/arm_motion/move_to_pose`
    comes back with a failure outcome (e.g. "planning_failed"), the
    orchestrator does NOT abort - it logs a warning, waits briefly, and
    requests a *fresh* perception estimate + grasp plan before trying
    again. The object is only abandoned (falling back to perception
    polling) when:
        - the retry budget is exhausted, or
        - the user explicitly declines a plan ("user_declined").

Once APPROACH + GRASP succeed for a given attempt, the sequence moves on
to the post-contact phase (closed-loop force grasp, lift, drop, release,
home). That phase is intentionally NOT part of the retry loop: once
contact has been made and/or the object has been lifted, re-requesting a
perception estimate and replanning from scratch is not meaningful (the
object may now be occluded by the hand, or physically moved). A failure
there ends the object cycle and returns to perception polling directly.

Interfaces
----------
Services:
    /perception/get_stable_estimate   (pcl_package/GetStableEstimate)
    /grasp_planning/get_grasp_pose    (my_package/GetGraspPose)
    /arm_motion/move_to_pose          (my_package/MoveToPose)

Topics:
    /preshape       (std_msgs/Int32)   - publish only, fire-and-forget
    /grasp_command  (std_msgs/String)  - publish only
    /grasp_status   (std_msgs/Int32)   - subscribe, 0 == success
"""

import threading

import rospy
from std_msgs.msg import Int32, String
from geometry_msgs.msg import Pose

from my_package.srv import (
    GetGraspPose, GetGraspPoseRequest,
    MoveToPose, MoveToPoseRequest,
)
from pcl_package.srv import GetStableEstimate, GetStableEstimateRequest


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Outcomes returned by /arm_motion/move_to_pose (mirrors MoveToPose.srv `outcome`)
MOVE_SUCCESS = "success"
MOVE_FAILED = "planning_failed"
MOVE_DECLINED = "user_declined"

# NOTE: PrimitiveEstimate.msg defines primitive_type as one of
# "SPHERE" / "CYLINDER" / "FLAT_BOX" / "UNKNOWN" (underscore, not "FLATBOX").
# These dictionaries are keyed EXACTLY on that string. A mismatch here
# silently drops a valid estimate with no error - always key off the
# message definition, never off ad-hoc spelling.
PRESHAPE_CODES = {
    "CYLINDER": 1,
    "SPHERE": 2,
    "FLAT_BOX": 3,
}
PRESHAPE_OPEN_CODE = 0  # sent during RELEASE to return the hand to its open pose

GRASP_STATUS_SUCCESS = 0


# --------------------------------------------------------------------------- #
# Small pose helper - kept free of tf/PyKDL so this file has no extra deps.
# --------------------------------------------------------------------------- #

def offset_pose(pose, dx=0.0, dy=0.0, dz=0.0):
    """Return a copy of `pose` translated by (dx, dy, dz) in the frame the
    pose is already expressed in (position only - orientation is preserved
    unchanged). Used for the vertical LIFT motion, where `grasp_pose_flange`
    is expressed in `ra_base_link` (world-vertical Z), so a plain Z addition
    gives a true vertical lift regardless of the flange's orientation."""
    new_pose = Pose()
    new_pose.position.x = pose.position.x + dx
    new_pose.position.y = pose.position.y + dy
    new_pose.position.z = pose.position.z + dz
    new_pose.orientation = pose.orientation
    return new_pose


def format_grasp_command(primitive_type, estimate):
    """Build the '<code>, <dims...>' string expected on /grasp_command,
    from the dimension fields populated in a PrimitiveEstimate."""
    code = PRESHAPE_CODES[primitive_type]
    if primitive_type == "CYLINDER":
        radius = estimate.diameter / 2.0
        return f"{code}, {radius:.5f}"
    if primitive_type == "SPHERE":
        radius = estimate.diameter / 2.0
        return f"{code}, {radius:.5f}"
    if primitive_type == "FLAT_BOX":
        half_width = estimate.width / 2.0
        half_thickness = estimate.thickness / 2.0
        return f"{code}, {half_width:.5f}, {half_thickness:.5f}"
    raise ValueError(f"Unsupported primitive_type '{primitive_type}' for /grasp_command")


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

class GraspingOrchestrator:
    """Drives one target object at a time through perception, planning,
    preshaping, motion, closed-loop grasping, and lift/drop/home. Callers
    are expected to invoke `run_object_cycle()` repeatedly from within a
    `while not rospy.is_shutdown()` outer loop (see `main()` below)."""

    def __init__(self):
        # ---- ROS params (all overridable, sensible defaults) --------------
        self._perception_poll_rate_hz = rospy.get_param("~perception_poll_rate_hz", 1.0)
        self._perception_timeout = rospy.get_param("~perception_timeout", 30.0)

        self._approach_wait_for_confirmation = rospy.get_param(
            "~approach_wait_for_confirmation", True
        )
        self._approach_velocity_scaling = rospy.get_param("~approach_velocity_scaling", 0.5)
        self._grasp_velocity_scaling = rospy.get_param("~grasp_velocity_scaling", 0.1)

        self._grasp_status_timeout = rospy.get_param("~grasp_status_timeout", 40.0)

        self._lift_height = rospy.get_param("~lift_height", 0.5)
        self._lift_velocity_scaling = rospy.get_param("~lift_velocity_scaling", 0.2)
        self._lift_dwell = rospy.get_param("~lift_dwell", 5.0)

        self._drop_pose = self._pose_from_param(
            "~drop_pose",
            default_xyz=(-0.011, 0.383, 0.965),
            default_xyzw=(0.742, 0.670, -0.014, 0.013),
        )
        self._home_pose = self._pose_from_param(
            "~home_pose",
            default_xyz=(0.228, 0.176, 1.436),
            default_xyzw=(-0.71, 0.011, 0.016, 0.704),
        )
        self._home_velocity_scaling = rospy.get_param("~home_velocity_scaling", 0.3)
        self._drop_velocity_scaling = rospy.get_param("~drop_velocity_scaling", 0.3)

        # ---- Nested retry-loop params (per target object) -----------------
        self._max_retries = rospy.get_param("~max_retries", 3)
        self._retry_delay = rospy.get_param("~retry_delay", 0.5)

        # ---- Service proxies -----------------------------------------------
        self._get_estimate = self._connect_service(
            "/perception/get_stable_estimate", GetStableEstimate
        )
        self._get_grasp = self._connect_service(
            "/grasp_planning/get_grasp_pose", GetGraspPose
        )
        self._move_to_pose = self._connect_service(
            "/arm_motion/move_to_pose", MoveToPose
        )

        # ---- Publishers ------------------------------------------------
        self._preshape_pub = rospy.Publisher("/preshape", Int32, queue_size=1)
        self._grasp_command_pub = rospy.Publisher("/grasp_command", String, queue_size=1)

        # ---- Grasp status subscriber (closed-loop grasp completion) -----
        self._grasp_status_event = threading.Event()
        self._last_grasp_status = None
        rospy.Subscriber("/grasp_status", Int32, self._on_grasp_status)

        rospy.loginfo("[grasping_orchestrator] Initialized and all services connected.")

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #

    def _connect_service(self, name, srv_type, timeout=10.0):
        try:
            rospy.loginfo(f"[grasping_orchestrator] Waiting for service '{name}'...")
            rospy.wait_for_service(name, timeout=timeout)
        except rospy.ROSException as e:
            rospy.logfatal(f"[grasping_orchestrator] Service '{name}' unavailable: {e}")
            raise
        return rospy.ServiceProxy(name, srv_type)

    @staticmethod
    def _pose_from_param(param_name, default_xyz, default_xyzw):
        """Reads {x,y,z,qx,qy,qz,qw} from a rosparam dict, falling back to
        the given defaults for any missing field."""
        d = rospy.get_param(param_name, {})
        pose = Pose()
        pose.position.x = d.get("x", default_xyz[0])
        pose.position.y = d.get("y", default_xyz[1])
        pose.position.z = d.get("z", default_xyz[2])
        pose.orientation.x = d.get("qx", default_xyzw[0])
        pose.orientation.y = d.get("qy", default_xyzw[1])
        pose.orientation.z = d.get("qz", default_xyzw[2])
        pose.orientation.w = d.get("qw", default_xyzw[3])
        return pose

    def _on_grasp_status(self, msg):
        self._last_grasp_status = msg.data
        if msg.data == GRASP_STATUS_SUCCESS:
            self._grasp_status_event.set()

    # ------------------------------------------------------------------ #
    # Candidate generation: perception + grasp planning
    # ------------------------------------------------------------------ #

    def _wait_for_object(self):
        """Poll /perception/get_stable_estimate at a fixed rate until a
        valid, stable estimate is returned or the timeout elapses.
        Returns (estimate, cloud) or (None, None) on failure/timeout."""
        rospy.loginfo("[grasping_orchestrator] Polling perception for a stable estimate...")
        rate = rospy.Rate(self._perception_poll_rate_hz)
        deadline = rospy.Time.now() + rospy.Duration(self._perception_timeout)

        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            try:
                res = self._get_estimate(GetStableEstimateRequest())
            except rospy.ServiceException as e:
                rospy.logerr(f"[grasping_orchestrator] Perception service call failed: {e}")
                return None, None

            if res.success and res.estimate.valid:
                rospy.loginfo(
                    f"[grasping_orchestrator] Got stable '{res.estimate.primitive_type}' estimate."
                )
                return res.estimate, res.cloud

            rospy.logwarn_throttle(
                2.0, f"[grasping_orchestrator] Waiting for stable estimate "
                     f"(reason: {res.reason or 'not yet valid'})..."
            )
            rate.sleep()

        rospy.logwarn("[grasping_orchestrator] Timed out waiting for a stable estimate.")
        return None, None

    def _plan_grasp(self, estimate, cloud):
        """Request a grasp candidate from /grasp_planning/get_grasp_pose.
        Returns the GetGraspPose response, or None on failure."""
        rospy.loginfo("[grasping_orchestrator] Requesting grasp candidate...")
        req = GetGraspPoseRequest()
        req.estimate = estimate
        req.cloud = cloud

        try:
            res = self._get_grasp(req)
        except rospy.ServiceException as e:
            rospy.logerr(f"[grasping_orchestrator] Grasp planning service call failed: {e}")
            return None

        if not res.success:
            rospy.logerr(f"[grasping_orchestrator] Grasp planning failed: {res.reason}")
            return None

        rospy.loginfo(
            f"[grasping_orchestrator] Grasp candidate accepted "
            f"(confidence={res.confidence:.2f}, method={res.confidence_method})."
        )
        return res

    def _acquire_and_plan(self):
        """Runs one perception + planning pass. Returns
        (estimate, cloud, grasp_response), any/all of which are None if
        that stage failed."""
        estimate, cloud = self._wait_for_object()
        if estimate is None:
            return None, None, None

        grasp_response = self._plan_grasp(estimate, cloud)
        if grasp_response is None:
            return None, None, None

        return estimate, cloud, grasp_response

    # ------------------------------------------------------------------ #
    # Preshape + arm motion for one candidate (retryable phase)
    # ------------------------------------------------------------------ #

    def _preshape(self, primitive_type):
        """Publish the preshape code for the classified primitive. Fire and
        forget - preshaping is instantaneous / non-blocking by design.
        Returns False if the primitive type has no known mapping."""
        if primitive_type not in PRESHAPE_CODES:
            rospy.logerr(
                f"[grasping_orchestrator] No preshape mapping for primitive_type "
                f"'{primitive_type}'."
            )
            return False

        code = PRESHAPE_CODES[primitive_type]
        rospy.loginfo(f"[grasping_orchestrator] PRESHAPE: publishing code {code} "
                       f"for '{primitive_type}'.")
        self._preshape_pub.publish(Int32(data=code))
        return True

    def _approach(self, grasp_response, estimate):
        """Move to the approach pose, with a collision object present
        (estimate passed through) so MoveIt plans around the target.
        Returns the move outcome string."""
        rospy.loginfo("[grasping_orchestrator] APPROACH: moving to approach_pose_flange...")
        return self._call_move_to_pose(
            target_pose=grasp_response.approach_pose_flange,
            estimate=estimate,
            wait_for_confirmation=self._approach_wait_for_confirmation,
            velocity_scaling=self._approach_velocity_scaling,
        )

    def _grasp(self, grasp_response):
        """Move to the final grasp pose. The collision object is omitted so
        MoveIt allows contact with the target, and confirmation is skipped
        since this motion is expected to run to completion automatically.
        Returns the move outcome string."""
        rospy.loginfo("[grasping_orchestrator] GRASP: moving to grasp_pose_flange...")
        return self._call_move_to_pose(
            target_pose=grasp_response.grasp_pose_flange,
            estimate=None,
            wait_for_confirmation=False,
            velocity_scaling=self._grasp_velocity_scaling,
        )

    def _execute_grasp_attempt(self, estimate, grasp_response):
        """Preshape + APPROACH + GRASP for one candidate. Returns
        MOVE_SUCCESS, MOVE_FAILED, or MOVE_DECLINED (preshape mapping
        failures are folded into MOVE_FAILED so they participate in the
        same retry logic as a failed arm motion)."""
        if not self._preshape(grasp_response.primitive_type):
            return MOVE_FAILED

        approach_outcome = self._approach(grasp_response, estimate)
        if approach_outcome != MOVE_SUCCESS:
            rospy.logwarn(
                f"[grasping_orchestrator] APPROACH outcome: '{approach_outcome}'."
            )
            return approach_outcome

        grasp_outcome = self._grasp(grasp_response)
        if grasp_outcome != MOVE_SUCCESS:
            rospy.logwarn(f"[grasping_orchestrator] GRASP outcome: '{grasp_outcome}'.")
        return grasp_outcome

    # ------------------------------------------------------------------ #
    # Post-contact phase (NOT retried - see module docstring)
    # ------------------------------------------------------------------ #

    def _closed_loop_grasp(self, estimate, grasp_response):
        """Trigger the closed-loop (force/proprioceptive) grasp controller
        via /grasp_command and block on /grasp_status == 0 up to a timeout."""
        primitive_type = grasp_response.primitive_type
        try:
            command_str = format_grasp_command(primitive_type, estimate)
        except ValueError as e:
            rospy.logerr(f"[grasping_orchestrator] {e}")
            return False

        rospy.loginfo(f"[grasping_orchestrator] CLOSED_LOOP_GRASP: publishing "
                       f"'/grasp_command' = \"{command_str}\".")
        self._grasp_status_event.clear()
        self._last_grasp_status = None
        self._grasp_command_pub.publish(String(data=command_str))

        got_success = self._grasp_status_event.wait(timeout=self._grasp_status_timeout)
        if not got_success:
            rospy.logerr(
                f"[grasping_orchestrator] Closed-loop grasp timed out after "
                f"{self._grasp_status_timeout}s (last status: {self._last_grasp_status})."
            )
            return False

        rospy.loginfo("[grasping_orchestrator] Closed-loop grasp reported success (status=0).")
        return True

    def _lift(self, grasp_response):
        """Lift straight up by `lift_height` from the grasp pose, then dwell."""
        lift_pose = offset_pose(grasp_response.grasp_pose_flange, dz=self._lift_height)
        rospy.loginfo(f"[grasping_orchestrator] LIFT: moving +{self._lift_height}m in Z...")
        outcome = self._call_move_to_pose(
            target_pose=lift_pose,
            estimate=None,
            wait_for_confirmation=True, #TODO : change this to false once we are confident in the grasping
            velocity_scaling=self._lift_velocity_scaling,
        )
        if outcome != MOVE_SUCCESS:
            rospy.logerr(f"[grasping_orchestrator] LIFT failed with outcome '{outcome}'.")
            return False

        rospy.loginfo(f"[grasping_orchestrator] Dwelling for {self._lift_dwell}s at lift height.")
        rospy.sleep(self._lift_dwell)
        return True

    def _drop(self):
        """Move to the pre-defined drop pose."""
        rospy.loginfo("[grasping_orchestrator] DROP: moving to configured drop pose...")
        outcome = self._call_move_to_pose(
            target_pose=self._drop_pose,
            estimate=None,
            wait_for_confirmation=True,  # TODO: change this to false once we are confident in the grasping
            velocity_scaling=self._drop_velocity_scaling,
        )
        if outcome != MOVE_SUCCESS:
            rospy.logerr(f"[grasping_orchestrator] DROP failed with outcome '{outcome}'.")
            return False
        return True

    def _release(self):
        """Open the hand by publishing the 'open' preshape code."""
        rospy.loginfo("[grasping_orchestrator] RELEASE: publishing open preshape code.")
        self._preshape_pub.publish(Int32(data=PRESHAPE_OPEN_CODE))
        rospy.sleep(0.5)  # brief settle time; preshaping is non-blocking on the topic side

    def _home(self):
        """Return the arm to the pre-defined home pose."""
        rospy.loginfo("[grasping_orchestrator] HOME: moving to configured home pose...")
        outcome = self._call_move_to_pose(
            target_pose=self._home_pose,
            estimate=None,
            wait_for_confirmation=True,  # TODO: change this to false once we are confident in the grasping
            velocity_scaling=self._home_velocity_scaling,
        )
        if outcome != MOVE_SUCCESS:
            rospy.logerr(f"[grasping_orchestrator] HOME failed with outcome '{outcome}'.")
            return False
        return True

    def _post_grasp_sequence(self, estimate, grasp_response):
        """CLOSED_LOOP_GRASP -> LIFT -> DROP -> RELEASE -> HOME. Runs once,
        after a successful APPROACH + GRASP - not part of the retry loop
        (see module docstring for rationale). Returns True only if every
        step completes; RELEASE always fires regardless so the hand isn't
        left closed on an early return."""
        if not self._closed_loop_grasp(estimate, grasp_response):
            return False
        if not self._lift(grasp_response):
            return False
        drop_ok = self._drop()
        self._release()
        if not drop_ok:
            return False
        return self._home()

    # ------------------------------------------------------------------ #
    # Shared service-call helper
    # ------------------------------------------------------------------ #

    def _call_move_to_pose(self, target_pose, estimate, wait_for_confirmation, velocity_scaling):
        """Wraps /arm_motion/move_to_pose, returning the outcome string
        (or MOVE_FAILED on a service-level exception)."""
        req = MoveToPoseRequest()
        req.target_pose = target_pose
        if estimate is not None:
            req.estimate = estimate
        req.wait_for_confirmation = wait_for_confirmation
        req.velocity_scaling = velocity_scaling

        try:
            res = self._move_to_pose(req)
        except rospy.ServiceException as e:
            rospy.logerr(f"[grasping_orchestrator] MoveToPose service call failed: {e}")
            return MOVE_FAILED

        if not res.success:
            rospy.logwarn(f"[grasping_orchestrator] MoveToPose reported failure: {res.reason}")
        return res.outcome

    # ------------------------------------------------------------------ #
    # Public entry point: one target object, with nested retry
    # ------------------------------------------------------------------ #

    def run_object_cycle(self):
        """Handles ONE target object end-to-end, including the nested
        candidate-generation + arm-motion retry loop described in the
        module docstring. Returns True if the full sequence (through
        HOME) completed successfully, False if the object was abandoned
        (retry budget exhausted or the user declined a plan)."""
        for attempt in range(1, self._max_retries + 1):
            rospy.loginfo(
                f"[grasping_orchestrator] --- Object attempt {attempt}/{self._max_retries} ---"
            )

            estimate, cloud, grasp_response = self._acquire_and_plan()
            if estimate is None:
                rospy.logwarn(
                    f"[grasping_orchestrator] Candidate generation failed on attempt "
                    f"{attempt}/{self._max_retries}."
                )
                if attempt < self._max_retries:
                    rospy.sleep(self._retry_delay)
                continue

            outcome = self._execute_grasp_attempt(estimate, grasp_response)

            if outcome == MOVE_DECLINED:
                rospy.loginfo(
                    "[grasping_orchestrator] User declined the plan; abandoning this object."
                )
                return False

            if outcome != MOVE_SUCCESS:
                rospy.logwarn(
                    f"[grasping_orchestrator] Arm motion failed ('{outcome}') on attempt "
                    f"{attempt}/{self._max_retries}; requesting a fresh candidate."
                )
                if attempt < self._max_retries:
                    rospy.sleep(self._retry_delay)
                continue

            # APPROACH + GRASP succeeded - move on to the (non-retried) post-contact phase.
            return self._post_grasp_sequence(estimate, grasp_response)

        rospy.logerr(
            f"[grasping_orchestrator] Abandoning object after {self._max_retries} failed attempts."
        )
        return False


# --------------------------------------------------------------------------- #
# Node entry point
# --------------------------------------------------------------------------- #

def main():
    rospy.init_node("grasping_orchestrator")

    inter_cycle_delay = rospy.get_param("~inter_cycle_delay", 1.0)

    try:
        orchestrator = GraspingOrchestrator()
    except rospy.ROSException:
        rospy.logfatal("[grasping_orchestrator] Aborting startup - a required service "
                        "never became available.")
        return

    rospy.loginfo("[grasping_orchestrator] Entering continuous perception polling loop.")
    while not rospy.is_shutdown():
        success = orchestrator.run_object_cycle()
        if success:
            rospy.loginfo(
                "[grasping_orchestrator] Object cycle complete - returning to perception polling."
            )
        else:
            rospy.logwarn(
                "[grasping_orchestrator] Object cycle ended without success - "
                "returning to perception polling."
            )

        if rospy.is_shutdown():
            break
        rospy.sleep(inter_cycle_delay)


if __name__ == "__main__":
    main()