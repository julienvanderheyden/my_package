#!/usr/bin/env python3
"""
cylinder_grasp_planner_node.py

Computes candidate rh_palm grasp poses for a cylindrical object, using the same
geometric logic as the Julia VMC virtual-cylinder placement code, and visualizes
the candidates in RViz as RGB axis triads.

------------------------------------------------------------------------------
IMPORTANT ASSUMPTIONS - PLEASE READ BEFORE TRUSTING THE OUTPUT
------------------------------------------------------------------------------
1. AXIS CONVENTION (the one thing most likely to be wrong for your URDF):
   This node assumes the cylinder's longitudinal axis, when the hand is
   correctly positioned to grasp it, is aligned with rh_forearm's LOCAL X AXIS.
   This is inferred from the provided Julia code (which only ever reads the Y
   and Z components of finger-frame origins, never X - implying the cylinder
   axis is X-invariant in that frame). This is almost certainly right in
   *structure*, but the exact sign/roll convention (R0 below) is a guess and
   should be verified visually in RViz: with theta=0, the palm's local axes
   should look sensible relative to the cylinder (approach axis wrapping
   around it, not pointing along it). Adjust R0 if the visualization looks
   wrong - flipping the sign of the -90 degree rotation, or swapping which
   local axis is used, are the first things to try.

2. FINGER GEOMETRY VIA LIVE TF, NOT THE JULIA PRESHAPE'S OWN FK:
   The Julia code computes finger-frame positions via forward kinematics at a
   SPECIFIC, FIXED preshape (medium_wrap_preshape, thumb extended, all other
   joints at 0). This node does NOT reimplement that FK - instead it looks up
   the CURRENT, LIVE tf2 transforms for rh_ffknuckle / rh_fftip / rh_ffmiddle /
   rh_thtip / rh_thmiddle relative to rh_forearm. This is only geometrically
   equivalent to the Julia computation if the real hand is ALREADY in (or is
   commanded to) that same medium-wrap preshape when this node queries tf2.
   The caller (your higher-level reaching node) is responsible for ensuring
   this - e.g. commanding the preshape before/while requesting candidate poses.

3. `circle_center_tangent_to_lines` IMPLEMENTATION:
   The Julia snippet calls this function but does not define it. This node
   reconstructs it as the standard "circle of given radius tangent to two
   lines" geometry problem: offset each line by `radius` toward the region
   between the two lines, then intersect the offset lines. This has been
   sanity-checked standalone (offsets land exactly `radius` from each original
   line), but it was NOT cross-validated against the original Julia function's
   exact output, since that implementation wasn't provided. If your grasp
   candidates look systematically off (e.g. circle appears on the wrong side
   of one of the fingers), this is the first place to check.

Both (1) and (3) are exactly what this node's RViz visualization is for -
inspect the candidate axis triads against the real cylinder and hand before
trusting this for actual motion planning.
------------------------------------------------------------------------------
"""

import rospy
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import tf2_ros
import tf2_geometry_msgs  

from geometry_msgs.msg import Pose, Point, Quaternion, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

from pcl_package.srv import GetStableEstimate

import moveit_commander
import sys


# ---------------------------------------------------------------------------
# Fixed frame names
# ---------------------------------------------------------------------------
EE_FRAME = "rh_palm"
HAND_BASE_FRAME = "rh_forearm"
INERTIAL_FRAME_DEFAULT = "world"
CAMERA_FRAME = "camera_color_optical_frame"

PALM_WIDTH = 0.084  # m

# Radius threshold from the Julia VMC code
SMALL_RADIUS_THRESHOLD = 0.015  # m

# Fixed clearance margins, matching the Julia code exactly
SMALL_RADIUS_CLEARANCE = 0.007  # m, subtracted for the small-radius branch
LARGE_RADIUS_CLEARANCE = 0.01   # m, added to radius for the tangent-circle branch


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def offset_line_2d(p_a, p_b, radius, towards_point):
    """
    Offsets the infinite line through p_a, p_b by `radius`, in the direction
    of whichever of the two perpendiculars points toward `towards_point`.
    Returns (point_on_offset_line, unit_direction_of_line).
    """
    d = p_b - p_a
    d = d / np.linalg.norm(d)
    normal = np.array([-d[1], d[0]])
    mid = (p_a + p_b) / 2.0
    if np.dot(normal, towards_point - mid) < 0:
        normal = -normal
    return p_a + normal * radius, d


def line_intersection_2d(p1, d1, p2, d2):
    """Intersects two 2D lines given as (point, direction). Returns None if parallel."""
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    b = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t, _s = np.linalg.solve(A, b)
    return p1 + t * d1


def circle_center_tangent_to_lines(p11, p12, p21, p22, radius):
    """
    Reconstruction of the Julia `circle_center_tangent_to_lines` function
    (not provided in the source snippet - see module docstring, point 3).

    Finds the center of a circle of the given radius, tangent to both lines
    (p11-p12) and (p21-p22), on the side of each line facing the other line
    (i.e. in the "wrap" region between the two fingers).
    """
    p11, p12, p21, p22 = (np.asarray(p, dtype=float) for p in (p11, p12, p21, p22))
    ref_toward_2 = (p21 + p22) / 2.0
    ref_toward_1 = (p11 + p12) / 2.0
    off1_pt, d1 = offset_line_2d(p11, p12, radius, ref_toward_2)
    off2_pt, d2 = offset_line_2d(p21, p22, radius, ref_toward_1)
    center = line_intersection_2d(off1_pt, d1, off2_pt, d2)
    if center is None:
        rospy.logwarn("circle_center_tangent_to_lines: offset lines are parallel, "
                       "cannot compute a unique tangent circle center.")
    return center


# Fixed rotation mapping HAND-frame axes to CYLINDER-local-frame axes, at the
# theta=0 reference orientation: maps hand's +X onto cylinder's +Z 
R0 = Rot.from_euler('y', -90, degrees=True).as_matrix()


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class CylinderGraspPlanner(object):
    def __init__(self):
        self.inertial_frame = rospy.get_param("~inertial_frame", INERTIAL_FRAME_DEFAULT)
        self.n_theta = rospy.get_param("~n_theta_samples", 12)
        self.n_axial = rospy.get_param("~n_axial_samples", 3)
        self.axis_marker_length = rospy.get_param("~axis_marker_length", 0.02)
        self.tf_timeout = rospy.Duration(rospy.get_param("~tf_timeout", 1.0))

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.marker_pub = rospy.Publisher(
            "/grasp_planning/cylinder_candidate_grasps", MarkerArray, queue_size=1, latch=True)

        rospy.wait_for_service("/perception/get_stable_estimate")
        self.perception_srv = rospy.ServiceProxy(
            "/perception/get_stable_estimate", GetStableEstimate)

        moveit_commander.roscpp_initialize(sys.argv)
        self.mgc = moveit_commander.MoveGroupCommander("right_arm")
        self.mgc.set_end_effector_link("rh_palm")

    # -- Perception -----------------------------------------------------
    def get_cylinder_estimate(self):
        """Calls the perception service, returns a PoseStamped + radius + height
        for the cylinder, transformed into the inertial frame - or None if not
        available / not a cylinder."""
        try:
            resp = self.perception_srv()
        except rospy.ServiceException as e:
            rospy.logerr("Perception service call failed: %s", e)
            return None

        if not resp.success:
            rospy.logwarn("Perception service reports no stable estimate: %s", resp.reason)
            return None

        if resp.estimate.primitive_type != "CYLINDER":
            rospy.logwarn("Stable estimate is type '%s', not CYLINDER - nothing to plan for.",
                           resp.estimate.primitive_type)
            return None

        pose_in = resp.estimate.pose  # PoseStamped, in camera_color_optical_frame

        try:
            pose_inertial = self.tf_buffer.transform(
                pose_in, self.inertial_frame, timeout=self.tf_timeout)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Failed to transform cylinder pose from %s to %s: %s",
                          pose_in.header.frame_id, self.inertial_frame, e)
            return None

        return {
            "pose": pose_inertial,
            "radius": resp.estimate.diameter / 2.0 + 2.0, #inflate radius to avoid collision with the object
            "height": resp.estimate.height + 30.0, # increase height to avoid collision with the object
        }

    # -- VMC-derived hand-relative cylinder offset -----------------------
    def lookup_forearm_relative(self, child_frame):
        """Looks up the CURRENT transform of `child_frame` relative to
        HAND_BASE_FRAME (rh_forearm). See assumption 2 in the module docstring
        regarding the required hand preshape."""
        t = self.tf_buffer.lookup_transform(
            HAND_BASE_FRAME, child_frame, rospy.Time(0), self.tf_timeout)
        tr = t.transform.translation
        return np.array([tr.x, tr.y, tr.z])

    def compute_vmc_offset(self, radius):
        """
        Direct port of the Julia VMC cylinder-placement logic. Returns
        (y_off, z_off): the position, in rh_forearm's local frame, of a point
        on the cylinder's axis (X-component is irrelevant / assumed axis-
        aligned with rh_forearm's local X - see assumption 1).
        """
        if radius < SMALL_RADIUS_THRESHOLD:
            ffknuckle = self.lookup_forearm_relative("rh_ffknuckle")
            z_off = ffknuckle[2] - radius - SMALL_RADIUS_CLEARANCE
            y_off = -0.03
            return y_off, z_off
        else:
            fftip = self.lookup_forearm_relative("rh_fftip")
            ffmiddle = self.lookup_forearm_relative("rh_ffmiddle")
            thtip = self.lookup_forearm_relative("rh_thtip")
            thmiddle = self.lookup_forearm_relative("rh_thmiddle")

            p11 = fftip[1:3]
            p12 = ffmiddle[1:3]
            p21 = thtip[1:3]
            p22 = thmiddle[1:3]

            center = circle_center_tangent_to_lines(
                p11, p12, p21, p22, radius + LARGE_RADIUS_CLEARANCE)
            if center is None:
                return None
            return float(center[0]), float(center[1])

    # -- Candidate pose generation ----------------------------------------
    def generate_candidates(self, cyl_pose_stamped, radius, height):
        """
        Generates candidate rh_palm poses (as a list of geometry_msgs/Pose, in
        the inertial frame), sampling azimuthal angle around the cylinder axis
        and axial position along it, subject to the palm-width constraint.
        """
        offset = self.compute_vmc_offset(radius)
        if offset is None:
            rospy.logerr("Could not compute VMC hand-relative cylinder offset.")
            return []
        y_off, z_off = offset
        p_hand_axis_point = np.array([0.0, y_off, z_off])

        # Axial (along-cylinder) range, per spec: Z_cyl in [-H/2 + W/2, H/2 - W/2]
        half_span = height / 2.0 - PALM_WIDTH / 2.0
        if half_span < 0:
            rospy.logwarn("Cylinder height (%.3fm) is shorter than the palm width (%.3fm) - "
                           "no fully-contained axial position exists. Falling back to a single "
                           "candidate centered on the cylinder.", height, PALM_WIDTH)
            axial_positions = [0.0]
        elif self.n_axial <= 1:
            axial_positions = [0.0]
        else:
            axial_positions = list(np.linspace(-half_span, half_span, self.n_axial))

        theta_positions = list(np.linspace(0.0, 2 * np.pi, self.n_theta, endpoint=False))

        # Cylinder pose in inertial frame
        p_cyl = np.array([cyl_pose_stamped.pose.position.x,
                           cyl_pose_stamped.pose.position.y,
                           cyl_pose_stamped.pose.position.z])
        q_cyl = cyl_pose_stamped.pose.orientation
        R_cyl = Rot.from_quat([q_cyl.x, q_cyl.y, q_cyl.z, q_cyl.w]).as_matrix()

        # Current rh_forearm -> rh_palm transform (see class docstring notes)
        try:
            t_fp = self.tf_buffer.lookup_transform(
                HAND_BASE_FRAME, EE_FRAME, rospy.Time(0), self.tf_timeout)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Failed to look up %s -> %s: %s", HAND_BASE_FRAME, EE_FRAME, e)
            return []

        tr = t_fp.transform.translation
        q = t_fp.transform.rotation
        t_forearm_to_palm = np.array([tr.x, tr.y, tr.z])
        R_forearm_to_palm = Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

        candidates = []
        for theta in theta_positions:
            R_theta_cyl = Rot.from_euler('z', theta).as_matrix()  # rotation about cylinder's local Z
            R_local = R_theta_cyl @ R0  # rh_forearm orientation, in cylinder-local frame

            for a in axial_positions:
                # Solve for rh_forearm origin (in cylinder-local frame) such that
                # the known hand-frame axis point maps to (0, 0, a) in cylinder-local coords.
                t_local = np.array([0.0, 0.0, a]) - R_local @ p_hand_axis_point

                # Compose rh_forearm candidate pose into the inertial frame
                R_forearm_world = R_cyl @ R_local
                t_forearm_world = R_cyl @ t_local + p_cyl

                # Compose with the current forearm->palm transform to get the rh_palm candidate
                R_palm_world = R_forearm_world @ R_forearm_to_palm
                t_palm_world = R_forearm_world @ t_forearm_to_palm + t_forearm_world

                pose = Pose()
                pose.position = Point(*t_palm_world)
                q_palm = Rot.from_matrix(R_palm_world).as_quat()  # [x,y,z,w]
                pose.orientation = Quaternion(*q_palm)
                candidates.append(pose)

        return candidates

    # -- Visualization -----------------------------------------------------
    def make_axis_marker(self, pose, axis_index, marker_id, frame_id, stamp):
        """
        Builds one ARROW marker representing one local axis (0=X/red, 1=Y/green,
        2=Z/blue) of the given candidate pose. ARROW markers point along their
        own local +X by default, so Y/Z axes are represented by additionally
        rotating the marker's orientation.
        """
        colors = [ColorRGBA(1.0, 0.0, 0.0, 1.0),
                  ColorRGBA(0.0, 1.0, 0.0, 1.0),
                  ColorRGBA(0.0, 0.0, 1.0, 1.0)]
        # Extra rotation so the marker's local +X aligns with the candidate's
        # local Y or Z axis, respectively. Identity for X.
        extra_rot = [
            np.eye(3),
            Rot.from_euler('z', 90, degrees=True).as_matrix(),
            Rot.from_euler('y', -90, degrees=True).as_matrix(),
        ][axis_index]

        q = pose.orientation
        R_pose = Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        R_marker = R_pose @ extra_rot
        q_marker = Rot.from_matrix(R_marker).as_quat()

        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = "cylinder_grasp_candidates"
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.position = pose.position
        marker.pose.orientation = Quaternion(*q_marker)
        marker.scale = Vector3(self.axis_marker_length,
                                self.axis_marker_length * 0.15,
                                self.axis_marker_length * 0.15)
        marker.color = colors[axis_index]
        marker.lifetime = rospy.Duration(0)  # persistent, replaced on next publish
        return marker

    def publish_candidates(self, candidates, frame_id):
        marker_array = MarkerArray()
        stamp = rospy.Time.now()
        marker_id = 0
        for pose in candidates:
            for axis_index in range(3):
                marker_array.markers.append(
                    self.make_axis_marker(pose, axis_index, marker_id, frame_id, stamp))
                marker_id += 1
        self.marker_pub.publish(marker_array)
        rospy.loginfo("Published %d candidate grasp poses (%d markers) on "
                       "/grasp_planning/cylinder_candidate_grasps",
                       len(candidates), len(marker_array.markers))

    def filter_grasps_with_ik(self, candidates):
        valid_candidates = []
        for i, pose in enumerate(candidates):
            # Set target pose in inertial frame
            self.mgc.set_pose_target(pose, end_effector_link=EE_FRAME)
            #self.mgc.set_pose_target(pose)  # uses default end-effector link

            # Check if IK exists and plan path without executing
            plan_success, plan, planning_time, error_code = self.mgc.plan()

            if plan_success and len(plan.joint_trajectory.points) > 0:
                valid_candidates.append(pose)
                rospy.loginfo(f"Candidate {i}: VALID IK & Collision-free")
            else:
                rospy.logwarn(f"Candidate {i}: REJECTED (Unreachable or Collision), error code: {error_code.val}")

            self.mgc.clear_pose_targets()

        return valid_candidates


    # -- Top-level entry point ---------------------------------------------
    def run_once(self):
        estimate = self.get_cylinder_estimate()
        if estimate is None:
            return False

        candidates = self.generate_candidates(
            estimate["pose"], estimate["radius"], estimate["height"])
        if not candidates:
            rospy.logerr("No candidate grasp poses were generated.")
            return False

        valid_candidates = self.filter_grasps_with_ik(candidates)
        self.publish_candidates(valid_candidates, self.inertial_frame)
        return True


def main():
    rospy.init_node("cylinder_grasp_planner_node")
    planner = CylinderGraspPlanner()

    success = planner.run_once()
    if not success:
        rospy.logwarn("Initial candidate generation failed - node will keep running; "
                       "call run_once() again (e.g. via a future service/topic trigger) "
                       "once perception/tf data is available.")

    rospy.spin()


if __name__ == "__main__":
    main()