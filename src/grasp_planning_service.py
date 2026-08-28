#!/usr/bin/env python3
"""
grasp_planner_node.py

Single service node: given ANY stable primitive estimate (CYLINDER,
FLAT_BOX, or SPHERE) + its consensus point cloud (as produced by
/perception/get_stable_estimate), generates candidate rh_palm grasp poses,
filters them through IK/motion-planning feasibility, ranks the survivors,
and returns the selected grasp + approach poses (palm and flange) along
with a confidence metric.

Adding a fourth primitive means writing one build_estimate lambda, one
generate_candidates method, one score_candidates method, and one entry in
PRIMITIVES - not a whole new file.
"""

from collections import namedtuple
from copy import deepcopy
import math
import os

import rospy
import rospkg
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import tf2_ros
import tf2_geometry_msgs

import PyKDL
from kdl_parser_py import urdf as kdl_parser_urdf
from urdf_parser_py.urdf import URDF as URDFModel
try:
    import xacro
except ImportError:
    xacro = None

from geometry_msgs.msg import Pose, Point, Quaternion, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from sensor_msgs import point_cloud2 as pc2

from my_package.srv import GetGraspPose, GetGraspPoseResponse

import moveit_commander
import sys
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from moveit_msgs.msg import MoveItErrorCodes


# ---------------------------------------------------------------------------
# Fixed frame names
# ---------------------------------------------------------------------------
EE_FRAME = "rh_palm"
FLANGE_FRAME = "ra_flange"
HAND_BASE_FRAME = "rh_forearm"
INERTIAL_FRAME_DEFAULT = "world"

PALM_WIDTH = 0.084                    # m - cylinder axial candidate spacing
LATERAL_PINCH_PRESHAPE_WIDTH = 0.085  # m - box width candidate spacing

# Fixed clearance values/thresholds from VMC position offsets
SMALL_RADIUS_THRESHOLD = 0.015  # m
SMALL_RADIUS_CLEARANCE = 0.007  # m
LARGE_RADIUS_CLEARANCE = 0.01   # m
BALL_APPROACH_CLEARANCE = 0.021  # m

# Fixed rotations mapping HAND-frame axes to each primitive's local-frame
# axes at each planner's theta=0 / reference candidate orientation.
CYLINDER_R0 = Rot.from_euler('y', -90, degrees=True).as_matrix()
FLATBOX_R0 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
SPHERE_R0 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

# Per-primitive approach standoff, in the ra_flange's own local frame
# (see compute_approach_poses). Only the magnitude/axis differs; the
# offset-application logic itself is shared.
APPROACH_OFFSETS = {
    "CYLINDER": np.array([-0.15, 0.0, -0.05]),
    "FLAT_BOX": np.array([-0.12, 0.05, 0.0]),
    "SPHERE": np.array([0.0, 0.0, -0.12]),
}

MEDIUM_WRAP_PRESHAPE = {"rh_THJ4": 1.2}
HAND_PACKAGE_NAME = "my_package"
HAND_URDF_RELATIVE_PATH = "urdf/sr_hand_vm_compatible.urdf"


# ---------------------------------------------------------------------------
# Geometry helpers (module-level: no planner state needed)
# ---------------------------------------------------------------------------
def offset_line_2d(p_a, p_b, radius, towards_point):
    d = p_b - p_a
    d = d / np.linalg.norm(d)
    normal = np.array([-d[1], d[0]])
    mid = (p_a + p_b) / 2.0
    if np.dot(normal, towards_point - mid) < 0:
        normal = -normal
    return p_a + normal * radius, d


def line_intersection_2d(p1, d1, p2, d2):
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    b = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t, _s = np.linalg.solve(A, b)
    return p1 + t * d1


def circle_center_tangent_to_lines(p11, p12, p21, p22, radius):
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
    """Closest point between a ray and an infinite line (same frame). Ray
    parameter clamped to >= 0. Returns (p_ray, p_line)."""
    d1 = ray_dir / np.linalg.norm(ray_dir)
    d2 = line_dir / np.linalg.norm(line_dir)
    r = ray_origin - line_point
    b = np.dot(d1, d2)
    d = np.dot(d1, r)
    e = np.dot(d2, r)
    denom = 1.0 - b * b

    if abs(denom) < 1e-6:
        t_ray = 0.0
        t_line = e
    else:
        t_ray = (b * e - d) / denom
        t_line = (e - b * d) / denom

    t_ray = max(t_ray, 0.0)
    p_ray = ray_origin + t_ray * d1
    p_line = line_point + t_line * d2
    return p_ray, p_line


def closest_point_on_ray_to_point(ray_origin, ray_dir, point):
    """Closest point on a ray (parameter clamped to >= 0) to a fixed point."""
    d = ray_dir / np.linalg.norm(ray_dir)
    t = np.dot(point - ray_origin, d)
    t = max(t, 0.0)
    return ray_origin + t * d


def rotation_aligning_vectors(a, b):
    """Returns the minimal-angle rotation matrix R such that R @ a == b, for
    unit vectors a, b. Used by the sphere candidate generator to turn a
    single sampled approach direction into a full candidate orientation,
    with no free roll parameter."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)

    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis = axis / np.linalg.norm(axis)
        return Rot.from_rotvec(axis * math.pi).as_matrix()

    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def spherical_fibonacci_directions(n):
    """N roughly-uniformly-spaced unit vectors over the full sphere."""
    golden_ratio = (1.0 + 5.0 ** 0.5) / 2.0
    directions = []
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        r_xy = math.sqrt(max(0.0, 1.0 - z * z))
        theta = 2.0 * math.pi * i / golden_ratio
        directions.append(np.array([r_xy * math.cos(theta), r_xy * math.sin(theta), z]))
    return directions


def is_crazy_plan(plan):
    n_points = len(plan.joint_trajectory.points)
    if n_points <= 0:
        return True
    traj = np.array([p.positions for p in plan.joint_trajectory.points])
    joint_sweep = [round(math.degrees(v), 1) for v in (traj.max(axis=0) - traj.min(axis=0))]
    return any(sweep > 180.0 for sweep in joint_sweep)


# ---------------------------------------------------------------------------
# Hand FK (CYLINDER only - the medium-wrap offset depends on live fingertip
# geometry at the preshape; FLAT_BOX/SPHERE use closed-form Julia offsets)
# ---------------------------------------------------------------------------
class HandFKSolver(object):
    """Loads the Shadow Hand URDF once and computes forward kinematics at a
    fixed joint configuration (the grasp preshape), independent of the real
    hand's live state. Any joint not present in MEDIUM_WRAP_PRESHAPE is
    taken to be at 0 rad."""

    def __init__(self, package_name=HAND_PACKAGE_NAME,
                 urdf_relative_path=HAND_URDF_RELATIVE_PATH,
                 preshape_overrides=None):
        self.preshape_overrides = dict(preshape_overrides
                                        if preshape_overrides is not None
                                        else MEDIUM_WRAP_PRESHAPE)

        urdf_path = self._resolve_urdf_path(package_name, urdf_relative_path)
        urdf_xml = self._load_urdf_xml(urdf_path)

        self.robot = URDFModel.from_xml_string(urdf_xml)
        ok, self.tree = kdl_parser_urdf.treeFromUrdfModel(self.robot)
        if not ok:
            raise RuntimeError(f"Failed to build a KDL tree from URDF '{urdf_path}'.")

        self._chain_cache = {}
        self._frame_cache = {}
        rospy.loginfo("HandFKSolver: loaded '%s' (preshape overrides: %s)",
                       urdf_path, self.preshape_overrides)

    @staticmethod
    def _resolve_urdf_path(package_name, relative_path):
        pkg_path = rospkg.RosPack().get_path(package_name)
        full_path = os.path.join(pkg_path, relative_path)
        if os.path.isfile(full_path):
            return full_path

        candidates = [full_path + ".xacro"] if not full_path.endswith(".xacro") \
            else [full_path[:-len(".xacro")]]
        for alt in candidates:
            if os.path.isfile(alt):
                return alt

        raise IOError(f"Could not find hand URDF at '{full_path}' "
                       f"(also tried: {candidates}).")

    @staticmethod
    def _load_urdf_xml(urdf_path):
        if urdf_path.endswith(".xacro"):
            if xacro is None:
                raise RuntimeError(
                    f"'{urdf_path}' is a xacro file but the xacro package is not "
                    "importable in this environment.")
            doc = xacro.process_file(urdf_path)
            return doc.toprettyxml(indent="  ")
        with open(urdf_path, "r") as f:
            return f.read()

    def _get_chain(self, base_frame, tip_frame):
        key = (base_frame, tip_frame)
        chain = self._chain_cache.get(key)
        if chain is None:
            chain = self.tree.getChain(base_frame, tip_frame)
            self._chain_cache[key] = chain
        return chain

    def _joint_value(self, joint_name):
        return self.preshape_overrides.get(joint_name, 0.0)

    def get_frame(self, base_frame, tip_frame):
        """Pose of tip_frame expressed in base_frame at the preshape
        configuration, as (R, t) numpy arrays. Cached per (base, tip)."""
        key = (base_frame, tip_frame)
        cached = self._frame_cache.get(key)
        if cached is not None:
            return cached

        chain = self._get_chain(base_frame, tip_frame)
        n_joints = chain.getNrOfJoints()
        q = PyKDL.JntArray(n_joints)

        fixed_joint_type = getattr(PyKDL.Joint, "None")  # KDL's "no motion" joint type
        j = 0
        for i in range(chain.getNrOfSegments()):
            joint = chain.getSegment(i).getJoint()
            if joint.getType() != fixed_joint_type:
                q[j] = self._joint_value(joint.getName())
                j += 1

        fk_solver = PyKDL.ChainFkSolverPos_recursive(chain)
        frame = PyKDL.Frame()
        if fk_solver.JntToCart(q, frame) < 0:
            raise RuntimeError(f"KDL FK failed for chain '{base_frame}' -> '{tip_frame}'.")

        R = np.array([[frame.M[r, c] for c in range(3)] for r in range(3)])
        t = np.array([frame.p[0], frame.p[1], frame.p[2]])

        self._frame_cache[key] = (R, t)
        return R, t

    def get_translation(self, base_frame, tip_frame):
        _, t = self.get_frame(base_frame, tip_frame)
        return t


# ---------------------------------------------------------------------------
# Per-primitive config: bound methods + static parameters, filled in once
# all methods exist (end of GraspPlanner.__init__). Adding a primitive means
# adding one of these, not a new file/class.
# ---------------------------------------------------------------------------
PrimitiveConfig = namedtuple("PrimitiveConfig", [
    "estimate_fields",       # callable(PrimitiveEstimate msg) -> dict of extra estimate fields
    "generate_candidates",   # bound method(estimate dict) -> [{'palm': Pose, ...}, ...]
    "score_candidates",      # bound method(candidates, estimate) -> (best_candidate, score_info)
    "approach_offset",       # np.array(3,), local ra_flange-frame approach standoff
])


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class GraspPlanner(object):
    def __init__(self):
        self.inertial_frame = rospy.get_param("~inertial_frame", INERTIAL_FRAME_DEFAULT)
        self.axis_marker_length = rospy.get_param("~axis_marker_length", 0.02)
        self.tf_timeout = rospy.Duration(rospy.get_param("~tf_timeout", 1.0))

        # CYLINDER params
        self.n_theta = rospy.get_param("~n_theta_samples", 12)
        self.n_axial = rospy.get_param("~n_axial_samples", 3)
        self.min_sector_points = rospy.get_param("~min_sector_points", 50)
        self.sector_height_half_width = rospy.get_param("~sector_height_half_width", PALM_WIDTH / 2.0)
        self.sector_angle_half_width = math.radians(rospy.get_param("~sector_angle_half_width_deg", 45.0))
        self.cylinder_radius_inflation = rospy.get_param("~cylinder_radius_inflation", 0.01)

        # FLAT_BOX params
        self.n_box_width = rospy.get_param("~n_box_width_samples", 5)

        # SPHERE params
        self.n_directions = rospy.get_param("~n_direction_samples", 50)
        self.min_cap_points = rospy.get_param("~min_cap_points", 50)
        self.cap_half_angle = math.radians(rospy.get_param("~cap_half_angle_deg", 40.0))
        self.sphere_radius_inflation = rospy.get_param("~sphere_radius_inflation", 0.0)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.hand_fk = HandFKSolver(
            package_name=rospy.get_param("~hand_urdf_package", HAND_PACKAGE_NAME),
            urdf_relative_path=rospy.get_param("~hand_urdf_relative_path", HAND_URDF_RELATIVE_PATH))

        self.marker_pub = rospy.Publisher(
            "/grasp_planning/candidate_grasps", MarkerArray, queue_size=1, latch=True)
        self.best_marker_pub = rospy.Publisher(
            "/grasp_planning/best_grasp", MarkerArray, queue_size=1, latch=True)

        moveit_commander.roscpp_initialize(sys.argv)
        self.mgc = moveit_commander.MoveGroupCommander("right_arm")
        self.mgc.set_planning_time(0.5)

        rospy.wait_for_service('/compute_ik')
        self.ik_service = rospy.ServiceProxy('/compute_ik', GetPositionIK)

        self.PRIMITIVES = {
            "CYLINDER": PrimitiveConfig(
                estimate_fields=lambda e: {
                    "radius": e.diameter / 2.0 + self.cylinder_radius_inflation,# inflated, for candidate generation
                    "radius_measured": e.diameter / 2.0,                        # raw, for residual scoring
                    "height": e.height,
                },
                generate_candidates=self._generate_cylinder_candidates,
                score_candidates=self._score_cylinder_candidates,
                approach_offset=APPROACH_OFFSETS["CYLINDER"],
            ),
            "FLAT_BOX": PrimitiveConfig(
                estimate_fields=lambda e: {
                    "thickness": e.thickness,
                    "width": e.width,
                    "depth": e.depth,
                },
                generate_candidates=self._generate_flatbox_candidates,
                score_candidates=self._score_flatbox_candidates,
                approach_offset=APPROACH_OFFSETS["FLAT_BOX"],
            ),
            "SPHERE": PrimitiveConfig(
                estimate_fields=lambda e: {
                    "radius": e.diameter / 2.0 + self.sphere_radius_inflation,
                    "radius_measured": e.diameter / 2.0,
                },
                generate_candidates=self._generate_sphere_candidates,
                score_candidates=self._score_sphere_candidates,
                approach_offset=APPROACH_OFFSETS["SPHERE"],
            ),
        }

        self.grasp_service = rospy.Service(
            "/grasp_planning/get_grasp_pose", GetGraspPose, self.handle_get_grasp)
        rospy.loginfo("Grasp planner ready on /grasp_planning/get_grasp_pose "
                       "(primitives: %s)", ", ".join(self.PRIMITIVES.keys()))

    # =========================================================================
    # Shared: rigid frame-offset helpers (palm<->flange conversions)
    # =========================================================================
    def _lookup_rigid_offset(self, target_frame, source_frame):
        """Returns (R, t) such that a pose whose local frame is source_frame
        can be re-expressed with local frame = target_frame via:
            R_target_world = R_source_world * R
            t_target_world = t_source_world + R_source_world.apply(t)
        i.e. (R, t) is the pose of target_frame as seen from source_frame.
        Returns None on tf failure."""
        try:
            t_tf = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rospy.Time(0), self.tf_timeout)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"Failed to lookup transform {source_frame} -> {target_frame}: {e}")
            return None

        tr = t_tf.transform.translation
        q = t_tf.transform.rotation
        R_target_in_source = Rot.from_quat([q.x, q.y, q.z, q.w]).inv()
        t_target_in_source = -R_target_in_source.apply([tr.x, tr.y, tr.z])
        return R_target_in_source, t_target_in_source

    @staticmethod
    def _apply_rigid_offset(pose_source_world, R_target_in_source, t_target_in_source):
        p = np.array([pose_source_world.position.x, pose_source_world.position.y, pose_source_world.position.z])
        q = [pose_source_world.orientation.x, pose_source_world.orientation.y,
             pose_source_world.orientation.z, pose_source_world.orientation.w]
        R_source_world = Rot.from_quat(q)

        R_target_world = R_source_world * R_target_in_source
        t_target_world = p + R_source_world.apply(t_target_in_source)

        out = Pose()
        out.position = Point(*t_target_world)
        out.orientation = Quaternion(*R_target_world.as_quat())
        return out

    # =========================================================================
    # Shared: pose composition (the one piece of geometry common to ALL THREE
    # candidate generators - only R_local/target_local/p_hand_ref_point vary)
    # =========================================================================
    @staticmethod
    def _pose_stamped_to_Rt(pose_stamped):
        p = pose_stamped.pose.position
        q = pose_stamped.pose.orientation
        t = np.array([p.x, p.y, p.z])
        R = Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        return R, t

    def _lookup_forearm_to_palm(self):
        """(t, R) of rh_palm relative to rh_forearm, at the hand's current
        live state. Shared by every candidate generator. Returns None on
        tf failure."""
        try:
            t_fp = self.tf_buffer.lookup_transform(
                HAND_BASE_FRAME, EE_FRAME, rospy.Time(0), self.tf_timeout)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Failed to look up %s -> %s: %s", HAND_BASE_FRAME, EE_FRAME, e)
            return None
        tr = t_fp.transform.translation
        q = t_fp.transform.rotation
        return np.array([tr.x, tr.y, tr.z]), Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

    @staticmethod
    def _compose_forearm_candidate(R_object, t_object, R_local, target_local, p_hand_ref_point):
        """Given the object's world pose (R_object, t_object), a candidate
        forearm orientation expressed in the object's local frame (R_local),
        a target point in the object's local frame that the hand's
        reference point should land on, and that reference point itself (in
        forearm-local coordinates, p_hand_ref_point), solves for and returns
        the forearm's (R, t) in the world/inertial frame."""
        t_local = target_local - R_local @ p_hand_ref_point
        R_forearm_world = R_object @ R_local
        t_forearm_world = R_object @ t_local + t_object
        return R_forearm_world, t_forearm_world

    @staticmethod
    def _forearm_to_palm_pose(R_forearm_world, t_forearm_world, t_forearm_to_palm, R_forearm_to_palm):
        R_palm_world = R_forearm_world @ R_forearm_to_palm
        t_palm_world = R_forearm_world @ t_forearm_to_palm + t_forearm_world
        pose = Pose()
        pose.position = Point(*t_palm_world)
        pose.orientation = Quaternion(*Rot.from_matrix(R_palm_world).as_quat())
        return pose

    # =========================================================================
    # Shared: estimate handling (from request, not a service call)
    # =========================================================================
    def _transform_pose_to_inertial(self, pose_stamped, label):
        try:
            return self.tf_buffer.transform(pose_stamped, self.inertial_frame, timeout=self.tf_timeout)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Failed to transform %s pose from %s to %s: %s",
                          label, pose_stamped.header.frame_id, self.inertial_frame, e)
            return None

    def build_estimate(self, req, primitive_type, estimate_fields_fn):
        """Packages the request's estimate+cloud into the working dict used
        throughout planning: pose in the inertial frame (candidate
        generation) and pose/cloud in the cloud's native frame (scoring -
        the cloud itself is never transformed). primitive-specific fields
        (radius/height, thickness/width/depth, ...) are added on top via
        estimate_fields_fn, from PRIMITIVES[primitive_type]."""
        pose_cloud_frame = req.estimate.pose
        pose_inertial = self._transform_pose_to_inertial(pose_cloud_frame, primitive_type)
        if pose_inertial is None:
            return None

        estimate = {
            "pose": pose_inertial,
            "pose_cloud_frame": pose_cloud_frame,
            "cloud": req.cloud,
        }
        estimate.update(estimate_fields_fn(req.estimate))
        return estimate

    # =========================================================================
    # CYLINDER: VMC offset + candidate generation
    # =========================================================================
    def _compute_cylinder_vmc_offset(self, radius):
        if radius < SMALL_RADIUS_THRESHOLD:
            ffknuckle = self.hand_fk.get_translation(HAND_BASE_FRAME, "rh_ffknuckle")
            z_off = ffknuckle[2] - radius - SMALL_RADIUS_CLEARANCE
            y_off = -0.03
            return y_off, z_off
        else:
            fftip = self.hand_fk.get_translation(HAND_BASE_FRAME, "rh_fftip")
            ffmiddle = self.hand_fk.get_translation(HAND_BASE_FRAME, "rh_ffmiddle")
            thtip = self.hand_fk.get_translation(HAND_BASE_FRAME, "rh_thtip")
            thmiddle = self.hand_fk.get_translation(HAND_BASE_FRAME, "rh_thmiddle")

            center = circle_center_tangent_to_lines(
                fftip[1:3], ffmiddle[1:3], thtip[1:3], thmiddle[1:3], radius + LARGE_RADIUS_CLEARANCE)
            if center is None:
                return None
            return float(center[0]), float(center[1])

    def _generate_cylinder_candidates(self, estimate):
        radius, height = estimate["radius"], estimate["height"]
        offset = self._compute_cylinder_vmc_offset(radius)
        if offset is None:
            rospy.logerr("Could not compute VMC hand-relative cylinder offset.")
            return []
        y_off, z_off = offset
        p_hand_ref_point = np.array([0.0, y_off, z_off])

        half_span = height / 2.0 - PALM_WIDTH / 2.0
        if half_span < 0:
            rospy.logwarn("Cylinder height (%.3fm) is shorter than the palm width (%.3fm) - "
                           "falling back to a single candidate centered on the cylinder.",
                           height, PALM_WIDTH)
            axial_positions = [0.0]
        elif self.n_axial <= 1:
            axial_positions = [0.0]
        else:
            axial_positions = list(np.linspace(-0.5 * half_span, 0.5 * half_span, self.n_axial))
        theta_positions = list(np.linspace(0.0, 2 * np.pi, self.n_theta, endpoint=False))

        R_obj, t_obj = self._pose_stamped_to_Rt(estimate["pose"])
        lookup = self._lookup_forearm_to_palm()
        if lookup is None:
            return []
        t_forearm_to_palm, R_forearm_to_palm = lookup

        candidates = []
        for theta in theta_positions:
            R_local = Rot.from_euler('z', theta).as_matrix() @ CYLINDER_R0
            for a in axial_positions:
                target_local = np.array([0.0, 0.0, a])
                R_fw, t_fw = self._compose_forearm_candidate(R_obj, t_obj, R_local, target_local, p_hand_ref_point)
                pose = self._forearm_to_palm_pose(R_fw, t_fw, t_forearm_to_palm, R_forearm_to_palm)
                candidates.append({"palm": pose})

        return candidates

    # =========================================================================
    # FLAT_BOX: VMC offset + candidate generation
    # =========================================================================
    def _compute_box_vmc_offset(self, thickness, depth):
        """
        Direct port of the Julia VMC box-placement snippet:
            box_dimensions = [box_thickness, box_width, 0.1]
            box_position = SVector(0.042 + box_dimensions[1], -0.03, 0.32 + box_dimensions[3])
        """
        return np.array([0.042 + thickness / 2, -0.03, 0.32 + depth / 2])

    def _generate_flatbox_candidates(self, estimate):
        """
        Targets the face perpendicular to the depth axis with the LOWER
        depth coordinate (the face closest to the camera - box local frame
        convention: local X = width_dir, local Y = depth_dir, local Z =
        normal). Returns [{'palm': Pose, 'width_offset': float}, ...] - the
        width_offset is carried through the pipeline for _score_flatbox_
        candidates, in place of the point-cloud-based score CYLINDER/SPHERE
        use.
        """
        thickness, width, depth = estimate["thickness"], estimate["width"], estimate["depth"]
        p_hand_ref_point = self._compute_box_vmc_offset(thickness, depth)

        half_span = width / 2.0 - LATERAL_PINCH_PRESHAPE_WIDTH / 2.0
        if half_span < 0:
            rospy.logwarn("Box width (%.3fm) is shorter than the preshape width (%.3fm) - "
                           "no fully-contained width position exists. Falling back to a "
                           "single candidate centered on the box.", width, LATERAL_PINCH_PRESHAPE_WIDTH)
            width_positions = [0.0]
        elif self.n_box_width <= 1:
            width_positions = [0.0]
        else:
            width_positions = list(np.linspace(-0.5 * half_span, 0.5 * half_span, self.n_box_width))

        R_obj, t_obj = self._pose_stamped_to_Rt(estimate["pose"])
        lookup = self._lookup_forearm_to_palm()
        if lookup is None:
            return []
        t_forearm_to_palm, R_forearm_to_palm = lookup

        R_local = FLATBOX_R0  # single fixed orientation - the box is not rotationally symmetric

        candidates = []
        for w in width_positions:
            target_local = np.array([w, 0.0, 0.0])
            R_fw, t_fw = self._compose_forearm_candidate(R_obj, t_obj, R_local, target_local, p_hand_ref_point)
            pose = self._forearm_to_palm_pose(R_fw, t_fw, t_forearm_to_palm, R_forearm_to_palm)
            candidates.append({"palm": pose, "width_offset": float(w)})

        return candidates

    # =========================================================================
    # SPHERE: VMC offset + candidate generation
    # =========================================================================
    def _compute_ball_vmc_offset(self, radius):
        """
        Direct port of the Julia VMC ball-placement snippet:
            ball_position = SVector(0.0, -ball_radius - 0.021, 0.33)
        The ball center's position in the forearm's own local frame at the
        reference candidate orientation (forearm-to-ball, forearm-local
        coordinates). The clearance (0.021) is already folded in.
        """
        return np.array([0.02, -radius - BALL_APPROACH_CLEARANCE, 0.33])

    def _generate_sphere_candidates(self, estimate):
        """
        Samples approach directions over a Spherical Fibonacci lattice - a
        sphere has no privileged translational DOF beyond approach
        direction, so direction sampling is the only candidate-generation
        axis; unreachable ones are left for IK to reject. SPHERE_R0
        additionally re-orients the reference-direction-aligned frame to
        match the hand's actual approach convention (see
        rotation_aligning_vectors - it only fixes the axis the offset
        vector points along, not the hand's roll about it).
        """
        radius = estimate["radius"]
        p_hand_ref_point = self._compute_ball_vmc_offset(radius)
        ref_norm = np.linalg.norm(p_hand_ref_point)
        if ref_norm < 1e-9:
            rospy.logerr("Degenerate ball VMC offset (zero norm) - cannot derive an approach direction.")
            return []
        ref_direction = p_hand_ref_point / ref_norm

        directions = spherical_fibonacci_directions(self.n_directions)

        # A sphere's orientation is not physically meaningful; treat its
        # local frame as coincident with the inertial frame (identity
        # rotation), only translated to the fitted center.
        _, t_ball = self._pose_stamped_to_Rt(estimate["pose"])
        R_ball = np.eye(3)

        lookup = self._lookup_forearm_to_palm()
        if lookup is None:
            return []
        t_forearm_to_palm, R_forearm_to_palm = lookup

        target_local = np.zeros(3)  # always the ball's own center

        candidates = []
        for d in directions:
            R_local = SPHERE_R0 @ rotation_aligning_vectors(ref_direction, d)
            R_fw, t_fw = self._compose_forearm_candidate(R_ball, t_ball, R_local, target_local, p_hand_ref_point)
            pose = self._forearm_to_palm_pose(R_fw, t_fw, t_forearm_to_palm, R_forearm_to_palm)
            candidates.append({"palm": pose})

        return candidates

    # =========================================================================
    # Shared: RViz visualization
    # =========================================================================
    def make_axis_marker(self, pose, axis_index, marker_id, frame_id, stamp, ns):
        colors = [ColorRGBA(1.0, 0.0, 0.0, 1.0),
                  ColorRGBA(0.0, 1.0, 0.0, 1.0),
                  ColorRGBA(0.0, 0.0, 1.0, 1.0)]
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
        marker.lifetime = rospy.Duration(0)
        return marker

    def publish_candidates(self, poses, frame_id, ns):
        marker_array = MarkerArray()
        stamp = rospy.Time.now()
        marker_id = 0
        for pose in poses:
            for axis_index in range(3):
                marker_array.markers.append(
                    self.make_axis_marker(pose, axis_index, marker_id, frame_id, stamp, ns))
                marker_id += 1
        self.marker_pub.publish(marker_array)
        rospy.loginfo("Published %d candidate grasp poses (%d markers) on "
                       "/grasp_planning/candidate_grasps", len(poses), len(marker_array.markers))

    def publish_best_candidate(self, palm_pose, frame_id, ns):
        marker_array = MarkerArray()
        stamp = rospy.Time.now()
        for axis_index in range(3):
            marker_array.markers.append(
                self.make_axis_marker(palm_pose, axis_index, axis_index, frame_id, stamp, ns))
        self.best_marker_pub.publish(marker_array)

    # =========================================================================
    # Shared: IK / planning filtering (candidate = dict; unknown keys preserved)
    # =========================================================================
    def filter_grasps_with_ik(self, candidates, pose_key):
        valid_candidates = []
        for i, cand in enumerate(candidates):
            pose = cand[pose_key]
            req = GetPositionIKRequest()
            req.ik_request.group_name = "right_arm"
            req.ik_request.ik_link_name = FLANGE_FRAME
            req.ik_request.pose_stamped.header.frame_id = self.inertial_frame
            req.ik_request.pose_stamped.header.stamp = rospy.Time.now()
            req.ik_request.pose_stamped.pose = pose
            req.ik_request.avoid_collisions = True
            req.ik_request.timeout = rospy.Duration(0.1)

            try:
                res = self.ik_service(req)
                if res.error_code.val == MoveItErrorCodes.SUCCESS:
                    valid_candidates.append(cand)
                else:
                    rospy.logdebug(f"Candidate {i} ({pose_key}): REJECTED (Error {res.error_code.val})")
            except rospy.ServiceException as e:
                rospy.logerr(f"IK Service call failed: {e}")

        return valid_candidates

    def transform_candidates_palm_to_flange(self, palm_candidates):
        """Pairs each rh_palm candidate dict with its ra_flange target,
        preserving any extra keys already on the candidate (e.g.
        'width_offset')."""
        offset = self._lookup_rigid_offset(FLANGE_FRAME, EE_FRAME)
        if offset is None:
            return []
        R_flange_in_palm, t_flange_in_palm = offset

        out = []
        for cand in palm_candidates:
            new_cand = dict(cand)
            new_cand["flange"] = self._apply_rigid_offset(
                cand["palm"], R_flange_in_palm, t_flange_in_palm)
            out.append(new_cand)
        return out

    def compute_approach_poses(self, candidates, local_offset):
        """Adds 'approach_flange' (offset backward/sideways in the flange's
        local frame - a MoveIt/IK convenience standoff, primitive-specific
        via local_offset, see APPROACH_OFFSETS) and 'approach_palm' (the
        palm pose implied by that same approach_flange target). All other
        candidate keys are preserved."""
        offset = self._lookup_rigid_offset(EE_FRAME, FLANGE_FRAME)
        if offset is None:
            return []
        R_palm_in_flange, t_palm_in_flange = offset

        out_candidates = []
        for cand in candidates:
            pose = cand["flange"]
            approach_flange = deepcopy(pose)

            q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            R = Rot.from_quat(q)
            world_offset = R.apply(local_offset)

            approach_flange.position.x += world_offset[0]
            approach_flange.position.y += world_offset[1]
            approach_flange.position.z += world_offset[2]

            new_cand = dict(cand)
            new_cand["approach_flange"] = approach_flange
            new_cand["approach_palm"] = self._apply_rigid_offset(
                approach_flange, R_palm_in_flange, t_palm_in_flange)
            out_candidates.append(new_cand)

        return out_candidates

    def filter_grasps_with_plans(self, candidates):
        valid_candidates = []
        for i, cand in enumerate(candidates):
            pose = cand["approach_flange"]
            self.mgc.set_pose_target(pose)
            success, plan, planning_time, error_code = self.mgc.plan()
            if success and len(plan.joint_trajectory.points) > 0:
                if not is_crazy_plan(plan):
                    valid_candidates.append(cand)
                else:
                    rospy.logdebug(f"Candidate {i}: REJECTED (crazy plan detected)")
            else:
                rospy.logdebug(f"Candidate {i}: REJECTED (no valid plan found)")
        return valid_candidates

    # =========================================================================
    # Shared: point-cloud utilities
    # =========================================================================
    def transform_ray_to_frame(self, origin, direction, source_frame, target_frame):
        t = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), self.tf_timeout)
        tr = t.transform.translation
        q = t.transform.rotation
        R = Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        t_vec = np.array([tr.x, tr.y, tr.z])
        return R @ origin + t_vec, R @ direction

    def cloud_to_array(self, cloud_msg):
        pts = list(pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True))
        if not pts:
            return np.zeros((0, 3))
        return np.asarray(pts, dtype=float)

    # =========================================================================
    # CYLINDER + SPHERE: shared point-count + radius-residual selection
    # =========================================================================
    def _score_cylinder_region(self, estimate, ray_origin_world, ray_dir_world):
        """Axial band + angular sector around the point on the cylinder's
        surface the candidate's approach ray is heading toward."""
        cloud_frame = estimate["pose_cloud_frame"].header.frame_id
        ray_origin_cloud, ray_dir_cloud = self.transform_ray_to_frame(
            ray_origin_world, ray_dir_world, self.inertial_frame, cloud_frame)

        cyl_pose = estimate["pose_cloud_frame"].pose
        axis_point = np.array([cyl_pose.position.x, cyl_pose.position.y, cyl_pose.position.z])
        q = cyl_pose.orientation
        R_cyl = Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        axis_dir = R_cyl[:, 2]

        p_ray, p_axis = closest_point_ray_to_line(ray_origin_cloud, ray_dir_cloud, axis_point, axis_dir)
        axial_h = float(np.dot(p_axis - axis_point, axis_dir))

        radial_vec = p_ray - p_axis
        radial_local = R_cyl.T @ radial_vec
        theta_center = math.atan2(radial_local[1], radial_local[0])

        points = self.cloud_to_array(estimate["cloud"])
        if points.shape[0] == 0:
            return 0, None

        rel = points - axis_point
        h = rel @ axis_dir
        radial = rel - np.outer(h, axis_dir)
        r = np.linalg.norm(radial, axis=1)
        local_xy = radial @ R_cyl[:, :2]
        theta = np.arctan2(local_xy[:, 1], local_xy[:, 0])
        dtheta = np.angle(np.exp(1j * (theta - theta_center)))

        in_band = np.abs(h - axial_h) <= self.sector_height_half_width
        in_sector = np.abs(dtheta) <= self.sector_angle_half_width
        mask = in_band & in_sector

        n_points = int(np.sum(mask))
        if n_points == 0:
            return 0, None

        residual = float(np.sqrt(np.mean((r[mask] - estimate["radius_measured"]) ** 2)))
        return n_points, residual

    def _score_sphere_cap(self, estimate, ray_origin_world, ray_dir_world):
        """Angular cap (great-circle radius cap_half_angle) around the point
        on the sphere the candidate's approach ray is heading toward. No
        axial band needed - a sphere has no axis."""
        cloud_frame = estimate["pose_cloud_frame"].header.frame_id
        ray_origin_cloud, ray_dir_cloud = self.transform_ray_to_frame(
            ray_origin_world, ray_dir_world, self.inertial_frame, cloud_frame)

        ball_pose = estimate["pose_cloud_frame"].pose
        center = np.array([ball_pose.position.x, ball_pose.position.y, ball_pose.position.z])

        p_ray = closest_point_on_ray_to_point(ray_origin_cloud, ray_dir_cloud, center)
        contact_dir = p_ray - center
        contact_norm = np.linalg.norm(contact_dir)
        if contact_norm < 1e-9:
            contact_dir = -ray_dir_cloud / np.linalg.norm(ray_dir_cloud)
        else:
            contact_dir = contact_dir / contact_norm

        points = self.cloud_to_array(estimate["cloud"])
        if points.shape[0] == 0:
            return 0, None

        rel = points - center
        r = np.linalg.norm(rel, axis=1)
        valid = r > 1e-9
        if not np.any(valid):
            return 0, None

        point_dirs = np.zeros_like(rel)
        point_dirs[valid] = rel[valid] / r[valid, None]
        cos_angle = np.clip(point_dirs @ contact_dir, -1.0, 1.0)
        angular_dev = np.arccos(cos_angle)

        mask = valid & (angular_dev <= self.cap_half_angle)
        n_points = int(np.sum(mask))
        if n_points == 0:
            return 0, None

        residual = float(np.sqrt(np.mean((r[mask] - estimate["radius_measured"]) ** 2)))
        return n_points, residual

    def _select_best_by_radius_fit(self, candidates, estimate, region_scorer, min_points, method_name):
        """Shared selection logic for CYLINDER and SPHERE: scores every
        candidate via region_scorer(estimate, ray_origin, ray_dir) -> (n_points,
        residual), then picks the best-fitting candidate among those with
        enough supporting points (or, failing that, the one with the most
        points). Confidence is 1.0 - residual as a fraction of the
        measured radius, clamped to [0, 1]."""
        scored = []
        for cand in candidates:
            palm = cand["palm"]
            origin = np.array([palm.position.x, palm.position.y, palm.position.z])
            q = palm.orientation
            R_palm = Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            direction = R_palm[:, 2]  # palm local approach direction

            n_points, residual = region_scorer(estimate, origin, direction)
            scored.append({"candidate": cand, "n_points": n_points, "residual": residual})

        qualifying = [s for s in scored if s["n_points"] >= min_points]

        if qualifying:
            best = min(qualifying, key=lambda s: s["residual"])
            met = True
            rospy.loginfo("[%s] selected by residual fit (%d/%d candidates had >= %d points).",
                           method_name, len(qualifying), len(scored), min_points)
        else:
            best = max(scored, key=lambda s: s["n_points"])
            met = False
            rospy.logwarn("[%s] no candidate reached %d points - falling back to "
                           "the most-evidenced candidate (%d points).",
                           method_name, min_points, best["n_points"])

        radius = estimate["radius_measured"]
        residual = best["residual"] if best["residual"] is not None else 0.0
        confidence = float(np.clip(1.0 - residual / radius, 0.0, 1.0)) if (met and radius > 0) else 0.0

        score_info = {
            "method": method_name,
            "point_count": best["n_points"],
            "metric_value": residual,
            "met_confidence_threshold": met,
            "confidence": confidence,
        }
        return best["candidate"], score_info

    def _score_cylinder_candidates(self, candidates, estimate):
        return self._select_best_by_radius_fit(
            candidates, estimate, self._score_cylinder_region, self.min_sector_points, "sector_residual")

    def _score_sphere_candidates(self, candidates, estimate):
        return self._select_best_by_radius_fit(
            candidates, estimate, self._score_sphere_cap, self.min_cap_points, "cap_residual")

    # =========================================================================
    # FLAT_BOX: centering-based selection (no independent fit metric exists -
    # see the module docstring)
    # =========================================================================
    def _score_flatbox_candidates(self, candidates, estimate):
        """Picks the surviving candidate whose width_offset is closest to
        the box's centerline (0.0), since there is no point-cloud fit
        metric available for a flat face seen from one side (see the
        module docstring). met_confidence_threshold is always True here:
        there is no point-count gate to fail, only a ranking to apply."""
        best = min(candidates, key=lambda c: abs(c["width_offset"]))

        half_span = max(estimate["width"] / 2.0 - LATERAL_PINCH_PRESHAPE_WIDTH / 2.0, 1e-6)
        confidence = float(np.clip(1.0 - abs(best["width_offset"]) / half_span, 0.0, 1.0))

        score_info = {
            "method": "width_centering",
            "point_count": 0,
            "metric_value": abs(best["width_offset"]),
            "met_confidence_threshold": True,
            "confidence": confidence,
        }
        return best, score_info

    # =========================================================================
    # Service handler
    # =========================================================================
    def handle_get_grasp(self, req):
        res = GetGraspPoseResponse()
        primitive_type = req.estimate.primitive_type

        config = self.PRIMITIVES.get(primitive_type)
        if config is None:
            res.success = False
            res.reason = (f"No grasp planner registered for primitive_type '{primitive_type}' "
                           f"(known types: {list(self.PRIMITIVES.keys())}).")
            return res

        estimate = self.build_estimate(req, primitive_type, config.estimate_fields)
        if estimate is None:
            res.success = False
            res.reason = f"Failed to transform {primitive_type} pose into the inertial frame."
            return res

        candidates = config.generate_candidates(estimate)
        if not candidates:
            res.success = False
            res.reason = "No candidate grasp poses were generated."
            return res

        paired_candidates = self.transform_candidates_palm_to_flange(candidates)
        if not paired_candidates:
            res.success = False
            res.reason = "Failed to compute the rh_palm -> ra_flange transform."
            return res

        valid_candidates = self.filter_grasps_with_ik(paired_candidates, pose_key="flange")
        valid_candidates = self.compute_approach_poses(valid_candidates, config.approach_offset)
        if not valid_candidates:
            res.success = False
            res.reason = "Failed to compute approach poses (rh_palm -> ra_flange transform lookup failed)."
            return res

        valid_candidates = self.filter_grasps_with_ik(valid_candidates, pose_key="approach_flange")
        valid_candidates = self.filter_grasps_with_plans(valid_candidates)
        if not valid_candidates:
            res.success = False
            res.reason = "No candidate grasp pose survived IK/planning filtering."
            return res

        marker_ns = primitive_type.lower()
        self.publish_candidates([c["approach_flange"] for c in valid_candidates], self.inertial_frame, marker_ns)

        best_candidate, score_info = config.score_candidates(valid_candidates, estimate)
        self.publish_best_candidate(best_candidate["palm"], self.inertial_frame, f"best_{marker_ns}")

        res.success = True
        res.primitive_type = primitive_type
        res.grasp_pose_palm = best_candidate["palm"]
        res.grasp_pose_flange = best_candidate["flange"]
        res.approach_pose_palm = best_candidate["approach_palm"]
        res.approach_pose_flange = best_candidate["approach_flange"]
        res.confidence = score_info["confidence"]
        res.confidence_method = score_info["method"]
        res.evidence_point_count = score_info["point_count"]
        res.met_confidence_threshold = score_info["met_confidence_threshold"]

        return res


def main():
    rospy.init_node("grasp_planner_node")
    GraspPlanner()
    rospy.spin()


if __name__ == "__main__":
    main()