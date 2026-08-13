#!/usr/bin/env python3
"""
cylinder_grasp_planner_node.py

Computes candidate rh_palm grasp poses for a cylindrical object, using the same
geometric logic as the Julia VMC virtual-cylinder placement code, filters them
through IK/motion-planning feasibility, and ranks the survivors by how much
local point-cloud evidence supports the cylindrical assumption right where the
hand will actually make contact.

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

4. RANKING BY LOCAL CLOUD EVIDENCE, NOT PALM POSE:
   The palm frame is not the frame that contacts the object, and by the time
   candidates survive the IK/planning filters they only exist as `ra_flange`
   targets (a MoveIt/IK convenience frame with no guaranteed physical relation
   to the grasp-approach direction). Each candidate therefore carries its
   original palm-frame pose alongside its flange-frame pose all the way through
   the pipeline (see `Candidate` dict below), and ranking uses ONLY the palm
   pose: the palm's local +X axis (the same axis `generate_candidates` was
   built around) is treated as the approach ray, and the point where that ray
   passes closest to the cylinder's axis line is taken as the predicted first-
   contact point. This gives both the axial height AND the azimuthal sector to
   query in the consensus point cloud from a single geometric computation.

Both (1) and (3) are exactly what this node's RViz visualization is for -
inspect the candidate axis triads against the real cylinder and hand before
trusting this for actual motion planning.
------------------------------------------------------------------------------
"""

from copy import deepcopy
import math

import rospy
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import tf2_ros
import tf2_geometry_msgs

from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from sensor_msgs import point_cloud2 as pc2

from pcl_package.srv import GetStableEstimate

import moveit_commander
import sys
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from moveit_msgs.msg import MoveItErrorCodes


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


def closest_point_ray_to_line(ray_origin, ray_dir, line_point, line_dir):
    """
    Closest point between a ray (origin, direction, both in the SAME frame as
    line_point/line_dir) and an infinite line. Standard closest-point-between-
    two-lines formula, with the ray parameter clamped to >= 0 (the point we
    care about is where the hand is headed, not behind the palm).

    Returns (p_ray, p_line): the closest point on the ray and the closest
    point on the line, respectively.
    """
    d1 = ray_dir / np.linalg.norm(ray_dir)
    d2 = line_dir / np.linalg.norm(line_dir)
    r = ray_origin - line_point
    b = np.dot(d1, d2)
    d = np.dot(d1, r)
    e = np.dot(d2, r)
    denom = 1.0 - b * b

    if abs(denom) < 1e-6:
        # Ray (near-)parallel to the cylinder axis - shouldn't happen for a
        # sane radial wrap-grasp approach, but fall back gracefully rather
        # than dividing by ~0.
        t_ray = 0.0
        t_line = e
    else:
        t_ray = (b * e - d) / denom
        t_line = (e - b * d) / denom

    t_ray = max(t_ray, 0.0)
    p_ray = ray_origin + t_ray * d1
    p_line = line_point + t_line * d2
    return p_ray, p_line


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

        # -- Sector scoring / ranking params --
        self.min_sector_points = rospy.get_param("~min_sector_points", 15)
        self.sector_height_half_width = rospy.get_param(
            "~sector_height_half_width", PALM_WIDTH / 2.0)  # m
        self.sector_angle_half_width = math.radians(
            rospy.get_param("~sector_angle_half_width_deg", 30.0))

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.marker_pub = rospy.Publisher(
            "/grasp_planning/cylinder_candidate_grasps", MarkerArray, queue_size=1, latch=True)
        self.best_marker_pub = rospy.Publisher(
            "/grasp_planning/best_cylinder_grasp", MarkerArray, queue_size=1, latch=True)

        rospy.wait_for_service("/perception/get_stable_estimate")
        self.perception_srv = rospy.ServiceProxy(
            "/perception/get_stable_estimate", GetStableEstimate)

        moveit_commander.roscpp_initialize(sys.argv)
        self.mgc = moveit_commander.MoveGroupCommander("right_arm")
        self.mgc.set_planning_time(0.5)  # default is usually 5.0s

        rospy.wait_for_service('/compute_ik')
        self.ik_service = rospy.ServiceProxy('/compute_ik', GetPositionIK)

    # -- Perception -----------------------------------------------------
    def get_cylinder_estimate(self):
        """Calls the perception service, returns the cylinder geometry in both
        the inertial frame (for candidate generation) and the cloud's native
        frame (for sector scoring - the consensus cloud stays untransformed,
        we transform the small number of candidate rays into its frame
        instead), plus the raw consensus cloud itself. Returns None if not
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

        pose_cloud_frame = resp.estimate.pose  # PoseStamped, native frame (e.g. camera_color_optical_frame)

        try:
            pose_inertial = self.tf_buffer.transform(
                pose_cloud_frame, self.inertial_frame, timeout=self.tf_timeout)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Failed to transform cylinder pose from %s to %s: %s",
                          pose_cloud_frame.header.frame_id, self.inertial_frame, e)
            return None

        return {
            "pose": pose_inertial,
            "pose_cloud_frame": pose_cloud_frame,
            "cloud": resp.cloud,
            "radius": resp.estimate.diameter / 2.0 + 0.01,  # inflated, for candidate generation / collision margin
            "radius_measured": resp.estimate.diameter / 2.0,  
            "height": resp.estimate.height,
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
            # TODO : don't rely on the actual finger position, use pyBullet or other FK solver
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
            axial_positions = list(np.linspace(-0.75*half_span, 0.75*half_span, self.n_axial))

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
    def make_axis_marker(self, pose, axis_index, marker_id, frame_id, stamp, ns="cylinder_grasp_candidates"):
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
        marker.ns = ns
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

    def publish_best_candidate(self, palm_pose, frame_id):
        marker_array = MarkerArray()
        stamp = rospy.Time.now()
        for axis_index in range(3):
            marker_array.markers.append(
                self.make_axis_marker(palm_pose, axis_index, axis_index, frame_id, stamp,
                                       ns="best_cylinder_grasp"))
        self.best_marker_pub.publish(marker_array)

    # -- IK / planning filtering (candidate = dict, see run_once) ----------
    def filter_grasps_with_ik(self, candidates, pose_key):
        """Filters a list of candidate dicts, testing IK on candidate[pose_key].
        Candidates (with ALL their keys, e.g. 'palm') are preserved unmodified
        for those that pass."""
        valid_candidates = []

        for i, cand in enumerate(candidates):
            pose = cand[pose_key]
            req = GetPositionIKRequest()
            req.ik_request.group_name = "right_arm"
            req.ik_request.ik_link_name = "ra_flange"
            req.ik_request.pose_stamped.header.frame_id = self.inertial_frame
            req.ik_request.pose_stamped.header.stamp = rospy.Time.now()
            req.ik_request.pose_stamped.pose = pose
            req.ik_request.avoid_collisions = True

            req.ik_request.timeout = rospy.Duration(0.1)

            try:
                res = self.ik_service(req)
                if res.error_code.val == MoveItErrorCodes.SUCCESS:
                    valid_candidates.append(cand)
                    rospy.loginfo(f"Candidate {i} ({pose_key}): VALID")
                else:
                    rospy.logwarn(f"Candidate {i} ({pose_key}): REJECTED (Error {res.error_code.val})")
            except rospy.ServiceException as e:
                rospy.logerr(f"IK Service call failed: {e}")

        return valid_candidates

    def transform_candidates_palm_to_flange(self, palm_candidates):
        """
        Pairs each rh_palm candidate pose (geometry_msgs/Pose, world frame)
        with its corresponding ra_flange target pose (world frame), returned
        as a list of dicts: [{'palm': Pose, 'flange': Pose}, ...]. The palm
        pose is carried forward through the whole pipeline so later stages
        (ranking) can use it without having to invert the flange transform.
        """
        try:
            # Lookup transform from ra_flange to rh_palm
            t = self.tf_buffer.lookup_transform(
                "ra_flange", "rh_palm", rospy.Time(0), self.tf_timeout
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"Failed to lookup transform ra_flange -> rh_palm: {e}")
            return []

        # Extract translation vector (palm -> flange in palm local frame)
        # Note: lookup_transform("ra_flange", "rh_palm") gives the pose of rh_palm in ra_flange.
        # We invert this to get ra_flange in rh_palm frame.
        tr = t.transform.translation
        q = t.transform.rotation

        R_flange_in_palm = Rot.from_quat([q.x, q.y, q.z, q.w]).inv()
        t_flange_in_palm = -R_flange_in_palm.apply([tr.x, tr.y, tr.z])

        paired_candidates = []

        for pose_palm in palm_candidates:
            # Extract palm candidate pose in world frame
            p_palm_world = np.array([pose_palm.position.x, pose_palm.position.y, pose_palm.position.z])
            q_palm_world = [pose_palm.orientation.x, pose_palm.orientation.y, pose_palm.orientation.z, pose_palm.orientation.w]
            R_palm_world = Rot.from_quat(q_palm_world)

            # Compose: R_flange_world = R_palm_world * R_flange_in_palm
            R_flange_world = R_palm_world * R_flange_in_palm

            # Compose: t_flange_world = t_palm_world + R_palm_world * t_flange_in_palm
            t_flange_world = p_palm_world + R_palm_world.apply(t_flange_in_palm)

            # Build output Pose
            pose_flange = Pose()
            pose_flange.position = Point(*t_flange_world)
            q_flange = R_flange_world.as_quat()
            pose_flange.orientation = Quaternion(*q_flange)

            paired_candidates.append({"palm": pose_palm, "flange": pose_flange})

        return paired_candidates

    def compute_approach_poses(self, candidates):
        """Adds an 'approach_flange' key to each candidate dict: the flange
        pose offset backward along the FLANGE frame's local X. This offset is
        purely for MoveIt/IK convenience (a sensible-looking pregrasp standoff
        for the arm's planner) - it is NOT used for anything involving actual
        contact prediction, see module docstring point 4."""
        approach_distance = 0.12  # m
        local_offset = np.array([-approach_distance, 0.0, 0.0])

        out_candidates = []
        for cand in candidates:
            pose = cand["flange"]
            approach_pose = deepcopy(pose)

            q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            R = Rot.from_quat(q)

            world_offset = R.apply(local_offset)

            approach_pose.position.x += world_offset[0]
            approach_pose.position.y += world_offset[1]
            approach_pose.position.z += world_offset[2]

            new_cand = dict(cand)
            new_cand["approach_flange"] = approach_pose
            out_candidates.append(new_cand)

        return out_candidates

    def filter_grasps_with_plans(self, candidates):
        valid_candidates = []
        for i, cand in enumerate(candidates):
            pose = cand["approach_flange"]
            self.mgc.set_pose_target(pose)
            success, plan, planning_time, error_code = self.mgc.plan()
            if success and len(plan.joint_trajectory.points) > 0:
                if self.is_crazy_plan(plan):
                    rospy.logwarn(f"Candidate {i}: REJECTED (crazy plan detected)")
                else:
                    valid_candidates.append(cand)
                    rospy.loginfo(f"Candidate {i}: VALID (plan found with {len(plan.joint_trajectory.points)} points)")
            else:
                rospy.logwarn(f"Candidate {i}: REJECTED (no valid plan found)")
        return valid_candidates

    def is_crazy_plan(self, plan):
        n_points = len(plan.joint_trajectory.points)
        if n_points <= 0:
            return True  # No points, consider it crazy
        elif n_points > 0:
            traj = np.array([p.positions for p in plan.joint_trajectory.points])
            joint_sweep = [round(math.degrees(v), 1) for v in (traj.max(axis=0) - traj.min(axis=0))]
            if any(sweep > 180.0 for sweep in joint_sweep):
                return True
            else:
                return False

    # -- Sector scoring / ranking -------------------------------------------
    def transform_ray_to_frame(self, origin, direction, source_frame, target_frame):
        """Transforms a ray (point + direction, both numpy arrays) from
        source_frame to target_frame using the current tf tree. Direction is
        rotated only (no translation), since it represents a free vector."""
        t = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), self.tf_timeout)
        tr = t.transform.translation
        q = t.transform.rotation
        R = Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        t_vec = np.array([tr.x, tr.y, tr.z])
        origin_out = R @ origin + t_vec
        dir_out = R @ direction
        return origin_out, dir_out

    def cloud_to_array(self, cloud_msg):
        pts = list(pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True))
        if not pts:
            return np.zeros((0, 3))
        return np.asarray(pts, dtype=float)

    def score_candidate_sector(self, cyl_estimate, ray_origin_world, ray_dir_world):
        """
        Projects the palm's approach ray onto the cylinder axis to find the
        predicted first-contact point, then scores the consensus cloud in a
        band (axial) x sector (azimuthal) window around it:
          - n_points: raw evidence count (used as a threshold/gate)
          - residual: RMS deviation from the fitted radius within that window
                      (used to rank candidates that pass the threshold)
        """
        cloud_frame = cyl_estimate["pose_cloud_frame"].header.frame_id

        ray_origin_cloud, ray_dir_cloud = self.transform_ray_to_frame(
            ray_origin_world, ray_dir_world, self.inertial_frame, cloud_frame)

        cyl_pose = cyl_estimate["pose_cloud_frame"].pose
        axis_point = np.array([cyl_pose.position.x, cyl_pose.position.y, cyl_pose.position.z])
        q = cyl_pose.orientation
        R_cyl = Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        axis_dir = R_cyl[:, 2]  # cylinder's local Z = its long axis

        p_ray, p_axis = closest_point_ray_to_line(ray_origin_cloud, ray_dir_cloud, axis_point, axis_dir)
        axial_h = float(np.dot(p_axis - axis_point, axis_dir))

        radial_vec = p_ray - p_axis
        radial_local = R_cyl.T @ radial_vec
        theta_center = math.atan2(radial_local[1], radial_local[0])

        points = self.cloud_to_array(cyl_estimate["cloud"])
        if points.shape[0] == 0:
            return 0, None

        rel = points - axis_point
        h = rel @ axis_dir
        radial = rel - np.outer(h, axis_dir)
        r = np.linalg.norm(radial, axis=1)
        local_xy = radial @ R_cyl[:, :2]
        theta = np.arctan2(local_xy[:, 1], local_xy[:, 0])

        dtheta = np.angle(np.exp(1j * (theta - theta_center)))  # wrap to [-pi, pi]

        in_band = np.abs(h - axial_h) <= self.sector_height_half_width
        in_sector = np.abs(dtheta) <= self.sector_angle_half_width
        mask = in_band & in_sector

        n_points = int(np.sum(mask))
        if n_points == 0:
            return 0, None

        residual = float(np.sqrt(np.mean((r[mask] - cyl_estimate["radius_measured"]) ** 2)))
        return n_points, residual

    def select_best_grasp(self, candidates, cyl_estimate):
        """
        Scores every surviving candidate's sector, then selects:
          - among candidates with n_points >= self.min_sector_points, the one
            with the LOWEST residual (best-supported fit to the cylinder
            assumption where contact will occur), or
          - if NONE reach the point threshold, the candidate with the most
            sector points (most evidence, even if not enough to fully trust).
        """
        scored = []
        for cand in candidates:
            palm = cand["palm"]
            origin = np.array([palm.position.x, palm.position.y, palm.position.z])
            q = palm.orientation
            R_palm = Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            direction = R_palm[:, 2]  # palm's local +Z axis 

            n_points, residual = self.score_candidate_sector(cyl_estimate, origin, direction)
            scored.append({"candidate": cand, "n_points": n_points, "residual": residual})

        qualifying = [s for s in scored if s["n_points"] >= self.min_sector_points]

        if qualifying:
            best = min(qualifying, key=lambda s: s["residual"])
            rospy.loginfo("Selected grasp by sector residual fit (%d/%d candidates had >= %d sector points).",
                           len(qualifying), len(scored), self.min_sector_points)
        else:
            best = max(scored, key=lambda s: s["n_points"])
            rospy.logwarn("No candidate reached %d sector points - falling back to "
                          "the candidate with the most sector evidence (%d points).",
                          self.min_sector_points, best["n_points"])

        return best["candidate"], best

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

        paired_candidates = self.transform_candidates_palm_to_flange(candidates)
        if not paired_candidates:
            return False

        valid_candidates = self.filter_grasps_with_ik(paired_candidates, pose_key="flange")
        valid_candidates = self.compute_approach_poses(valid_candidates)
        valid_candidates = self.filter_grasps_with_ik(valid_candidates, pose_key="approach_flange")
        valid_candidates = self.filter_grasps_with_plans(valid_candidates)

        if not valid_candidates:
            rospy.logerr("No candidate grasp pose survived IK/planning filtering.")
            return False

        self.publish_candidates([c["approach_flange"] for c in valid_candidates], self.inertial_frame)

        best_candidate, score_info = self.select_best_grasp(valid_candidates, estimate)
        rospy.loginfo("Best grasp candidate: %d sector points, residual=%s",
                       score_info["n_points"],
                       f"{score_info['residual']:.4f}" if score_info["residual"] is not None else "N/A")
        self.publish_best_candidate(best_candidate["palm"], self.inertial_frame)

        # best_candidate["approach_flange"] / best_candidate["flange"] are the
        # poses to actually send to MoveIt for the pregrasp / final grasp motions.
        self.selected_candidate = best_candidate
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