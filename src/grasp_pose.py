#!/usr/bin/env python3
"""
cylinder_grasp_planner_node.py

Service node: given a stable CYLINDER primitive estimate + its consensus point
cloud (as produced by /perception/get_stable_estimate), generates candidate
rh_palm grasp poses, filters them through IK/motion-planning feasibility, and
ranks the survivors by how much local point-cloud evidence supports the
cylindrical assumption right where the hand will actually make contact.

"""

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

from my_package.srv import GetCylinderGraspPose, GetCylinderGraspPoseResponse

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

PALM_WIDTH = 0.084  # m

SMALL_RADIUS_THRESHOLD = 0.015  # m
SMALL_RADIUS_CLEARANCE = 0.007  # m
LARGE_RADIUS_CLEARANCE = 0.01   # m


# ---------------------------------------------------------------------------
# Geometry helpers
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


R0 = Rot.from_euler('y', -90, degrees=True).as_matrix()

# ---------------------------------------------------------------------------
# Preshape configuration: all hand joints at zero, except rh_THJ4 which is
# rotated to 1.2 rad. compute_vmc_offset() needs the hand geometry in this
# pose - not whatever pose the real hand currently happens to be in - so it
# is obtained via FK against the URDF rather than a live /tf lookup.
# ---------------------------------------------------------------------------
MEDIUM_WRAP_PRESHAPE = {"rh_THJ4": 1.2}
HAND_PACKAGE_NAME = "my_package"
HAND_URDF_RELATIVE_PATH = "urdf/sr_hand_vm_compatible.urdf"


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

        # Be forgiving about the .urdf vs .urdf.xacro naming, since both are
        # common for the Shadow hand description.
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
        configuration, as (R, t) numpy arrays (3x3 rotation, 3-vector).
        Cached per (base, tip): the preshape never changes between calls,
        so there is no reason to re-solve the same chain every request."""
        key = (base_frame, tip_frame)
        cached = self._frame_cache.get(key)
        if cached is not None:
            return cached

        chain = self._get_chain(base_frame, tip_frame)
        n_joints = chain.getNrOfJoints()
        q = PyKDL.JntArray(n_joints)

        fixed_joint_type = getattr(PyKDL.Joint, "None")  # KDL calls "no motion" joints "None"
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
# Node
# ---------------------------------------------------------------------------
class CylinderGraspPlanner(object):
    def __init__(self):
        self.inertial_frame = rospy.get_param("~inertial_frame", INERTIAL_FRAME_DEFAULT)
        self.n_theta = rospy.get_param("~n_theta_samples", 12)
        self.n_axial = rospy.get_param("~n_axial_samples", 3)
        self.axis_marker_length = rospy.get_param("~axis_marker_length", 0.02)
        self.tf_timeout = rospy.Duration(rospy.get_param("~tf_timeout", 1.0))

        self.min_sector_points = rospy.get_param("~min_sector_points", 50)
        self.sector_height_half_width = rospy.get_param(
            "~sector_height_half_width", PALM_WIDTH / 2.0)
        self.sector_angle_half_width = math.radians(
            rospy.get_param("~sector_angle_half_width_deg", 45.0))

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.hand_fk = HandFKSolver(
            package_name=rospy.get_param("~hand_urdf_package", HAND_PACKAGE_NAME),
            urdf_relative_path=rospy.get_param("~hand_urdf_relative_path", HAND_URDF_RELATIVE_PATH))

        self.marker_pub = rospy.Publisher(
            "/grasp_planning/cylinder_candidate_grasps", MarkerArray, queue_size=1, latch=True)
        self.best_marker_pub = rospy.Publisher(
            "/grasp_planning/best_cylinder_grasp", MarkerArray, queue_size=1, latch=True)

        moveit_commander.roscpp_initialize(sys.argv)
        self.mgc = moveit_commander.MoveGroupCommander("right_arm")
        self.mgc.set_planning_time(0.5)

        rospy.wait_for_service('/compute_ik')
        self.ik_service = rospy.ServiceProxy('/compute_ik', GetPositionIK)

        self.grasp_service = rospy.Service(
            "/grasp_planning/get_cylinder_grasp", GetCylinderGraspPose, self.handle_get_grasp)
        rospy.loginfo("Cylinder grasp planner ready on /grasp_planning/get_cylinder_grasp")

    # -- Rigid frame-offset helpers (used for palm<->flange conversions) ---
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

    # -- Estimate handling (from request, not a service call) --------------
    def build_cylinder_estimate(self, req):
        """Packages the request's estimate+cloud into the working dict used
        throughout planning: pose in the inertial frame (for candidate
        generation) and pose/cloud in the cloud's native frame (for sector
        scoring - the cloud itself is never transformed, only small rays are)."""
        pose_cloud_frame = req.estimate.pose  # PoseStamped, native perception frame

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
            "cloud": req.cloud,
            "radius": req.estimate.diameter / 2.0 + 0.02,       # inflated, for candidate generation
            "radius_measured": req.estimate.diameter / 2.0,     # raw, for residual scoring
            "height": req.estimate.height,
        }

    # -- VMC-derived hand-relative cylinder offset -----------------------
    def lookup_forearm_relative(self, child_frame):
        """Translation of child_frame relative to HAND_BASE_FRAME, computed
        via FK against the URDF at the fixed preshape configuration (all
        joints at 0 rad, rh_THJ4 = 1.2 rad) - NOT the live /tf pose of the
        real hand, which may not be in preshape when this is called."""
        return self.hand_fk.get_translation(HAND_BASE_FRAME, child_frame)

    def compute_vmc_offset(self, radius):
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
        offset = self.compute_vmc_offset(radius)
        if offset is None:
            rospy.logerr("Could not compute VMC hand-relative cylinder offset.")
            return []
        y_off, z_off = offset
        p_hand_axis_point = np.array([0.0, y_off, z_off])

        half_span = height / 2.0 - PALM_WIDTH / 2.0
        if half_span < 0:
            rospy.logwarn("Cylinder height (%.3fm) is shorter than the palm width (%.3fm) - "
                           "falling back to a single candidate centered on the cylinder.",
                           height, PALM_WIDTH)
            axial_positions = [0.0]
        elif self.n_axial <= 1:
            axial_positions = [0.0]
        else:
            axial_positions = list(np.linspace(-0.75 * half_span, 0.75 * half_span, self.n_axial))

        theta_positions = list(np.linspace(0.0, 2 * np.pi, self.n_theta, endpoint=False))

        p_cyl = np.array([cyl_pose_stamped.pose.position.x,
                           cyl_pose_stamped.pose.position.y,
                           cyl_pose_stamped.pose.position.z])
        q_cyl = cyl_pose_stamped.pose.orientation
        R_cyl = Rot.from_quat([q_cyl.x, q_cyl.y, q_cyl.z, q_cyl.w]).as_matrix()

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
            R_theta_cyl = Rot.from_euler('z', theta).as_matrix()
            R_local = R_theta_cyl @ R0

            for a in axial_positions:
                t_local = np.array([0.0, 0.0, a]) - R_local @ p_hand_axis_point

                R_forearm_world = R_cyl @ R_local
                t_forearm_world = R_cyl @ t_local + p_cyl

                R_palm_world = R_forearm_world @ R_forearm_to_palm
                t_palm_world = R_forearm_world @ t_forearm_to_palm + t_forearm_world

                pose = Pose()
                pose.position = Point(*t_palm_world)
                q_palm = Rot.from_matrix(R_palm_world).as_quat()
                pose.orientation = Quaternion(*q_palm)
                candidates.append(pose)

        return candidates

    # -- Visualization -----------------------------------------------------
    def make_axis_marker(self, pose, axis_index, marker_id, frame_id, stamp, ns="cylinder_grasp_candidates"):
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

    # -- IK / planning filtering --------------------------------------------
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
        """Pairs each rh_palm candidate with its ra_flange target, returned as
        [{'palm': Pose, 'flange': Pose}, ...]. Palm is carried forward for
        ranking; see module docstring point 4."""
        offset = self._lookup_rigid_offset(FLANGE_FRAME, EE_FRAME)
        if offset is None:
            return []
        R_flange_in_palm, t_flange_in_palm = offset

        return [{"palm": pose_palm,
                 "flange": self._apply_rigid_offset(pose_palm, R_flange_in_palm, t_flange_in_palm)}
                for pose_palm in palm_candidates]

    def compute_approach_poses(self, candidates):
        """Adds 'approach_flange' (offset backward along the flange's local X,
        a MoveIt/IK convenience standoff) and 'approach_palm' (the palm pose
        implied by that same approach_flange target - see module docstring
        point 5, NOT an independent palm-frame offset)."""
        approach_distance = 0.15  # m
        local_offset = np.array([-approach_distance, 0.0, 0.0])

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
                if not self.is_crazy_plan(plan):
                    valid_candidates.append(cand)
                else:
                    rospy.logdebug(f"Candidate {i}: REJECTED (crazy plan detected)")
            else:
                rospy.logdebug(f"Candidate {i}: REJECTED (no valid plan found)")
        return valid_candidates

    def is_crazy_plan(self, plan):
        n_points = len(plan.joint_trajectory.points)
        if n_points <= 0:
            return True
        traj = np.array([p.positions for p in plan.joint_trajectory.points])
        joint_sweep = [round(math.degrees(v), 1) for v in (traj.max(axis=0) - traj.min(axis=0))]
        return any(sweep > 180.0 for sweep in joint_sweep)

    # -- Sector scoring / ranking -------------------------------------------
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

    def score_candidate_sector(self, cyl_estimate, ray_origin_world, ray_dir_world):
        cloud_frame = cyl_estimate["pose_cloud_frame"].header.frame_id

        ray_origin_cloud, ray_dir_cloud = self.transform_ray_to_frame(
            ray_origin_world, ray_dir_world, self.inertial_frame, cloud_frame)

        cyl_pose = cyl_estimate["pose_cloud_frame"].pose
        axis_point = np.array([cyl_pose.position.x, cyl_pose.position.y, cyl_pose.position.z])
        q = cyl_pose.orientation
        R_cyl = Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        axis_dir = R_cyl[:, 2]

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
        dtheta = np.angle(np.exp(1j * (theta - theta_center)))

        in_band = np.abs(h - axial_h) <= self.sector_height_half_width
        in_sector = np.abs(dtheta) <= self.sector_angle_half_width
        mask = in_band & in_sector

        n_points = int(np.sum(mask))
        if n_points == 0:
            return 0, None

        residual = float(np.sqrt(np.mean((r[mask] - cyl_estimate["radius_measured"]) ** 2)))
        return n_points, residual

    def select_best_grasp(self, candidates, cyl_estimate):
        scored = []
        for cand in candidates:
            palm = cand["palm"]
            origin = np.array([palm.position.x, palm.position.y, palm.position.z])
            q = palm.orientation
            R_palm = Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            direction = R_palm[:, 2] # palm local approach direction

            n_points, residual = self.score_candidate_sector(cyl_estimate, origin, direction)
            scored.append({"candidate": cand, "n_points": n_points, "residual": residual})

        qualifying = [s for s in scored if s["n_points"] >= self.min_sector_points]

        if qualifying:
            best = min(qualifying, key=lambda s: s["residual"])
            best["met_confidence_threshold"] = True
            rospy.loginfo("Selected grasp by sector residual fit (%d/%d candidates had >= %d sector points).",
                           len(qualifying), len(scored), self.min_sector_points)
        else:
            best = max(scored, key=lambda s: s["n_points"])
            best["met_confidence_threshold"] = False
            rospy.logwarn("No candidate reached %d sector points - falling back to "
                          "the candidate with the most sector evidence (%d points).",
                          self.min_sector_points, best["n_points"])

        return best["candidate"], best

    # -- Service handler -----------------------------------------------------
    def handle_get_grasp(self, req):
        res = GetCylinderGraspPoseResponse()

        if req.primitive_type != "CYLINDER" or req.estimate.primitive_type != "CYLINDER":
            res.success = False
            res.reason = "Request primitive_type is not CYLINDER."
            return res

        cyl_estimate = self.build_cylinder_estimate(req)
        if cyl_estimate is None:
            res.success = False
            res.reason = "Failed to transform cylinder pose into the inertial frame."
            return res

        candidates = self.generate_candidates(
            cyl_estimate["pose"], cyl_estimate["radius"], cyl_estimate["height"])
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
        valid_candidates = self.compute_approach_poses(valid_candidates)
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

        self.publish_candidates([c["approach_flange"] for c in valid_candidates], self.inertial_frame)

        best_candidate, score_info = self.select_best_grasp(valid_candidates, cyl_estimate)
        self.publish_best_candidate(best_candidate["palm"], self.inertial_frame)

        res.success = True
        res.grasp_pose_palm = best_candidate["palm"]
        res.grasp_pose_flange = best_candidate["flange"]
        res.approach_pose_palm = best_candidate["approach_palm"]
        res.approach_pose_flange = best_candidate["approach_flange"]
        res.sector_point_count = score_info["n_points"]
        res.sector_residual = score_info["residual"] if score_info["residual"] is not None else 0.0
        res.met_confidence_threshold = score_info["met_confidence_threshold"]

        return res


def main():
    rospy.init_node("cylinder_grasp_planner_node")
    CylinderGraspPlanner()
    rospy.spin()


if __name__ == "__main__":
    main()