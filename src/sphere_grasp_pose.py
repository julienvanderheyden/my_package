#!/usr/bin/env python3
"""
sphere_grasp_planner_node.py

Service node: given a stable SPHERE primitive estimate + its consensus point
cloud (as produced by /perception/get_stable_estimate and forwarded by the
grasp-planning dispatcher, exactly as for the cylinder and flat-box
planners), generates candidate rh_palm grasp poses by sampling approach
directions over a Spherical Fibonacci lattice, filters them through
IK/motion-planning feasibility, and ranks the survivors by how much local
point-cloud evidence supports the fitted radius right where the hand will
actually make contact - the same point-count + residual strategy the
cylinder planner uses, adapted from an axial band + angular sector to a
single angular cap (a sphere has no axis to band around).
"""

from copy import deepcopy
import math

import rospy
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import tf2_ros
import tf2_geometry_msgs

from geometry_msgs.msg import Pose, Point, Quaternion, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from sensor_msgs import point_cloud2 as pc2

from my_package.srv import GetSphereGraspPose, GetSphereGraspPoseResponse

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

# Fixed clearance baked into the VMC ball offset below (see
# compute_ball_vmc_offset) - kept as a named constant purely for
# documentation; it is not added on top of anything, it IS the 0.021 term.
BALL_APPROACH_CLEARANCE = 0.021  # m


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def rotation_aligning_vectors(a, b):
    """Returns the minimal-angle rotation matrix R such that R @ a == b, for
    unit vectors a, b. Used to turn a single sampled approach direction into
    a full candidate orientation: it rotates the hand's fixed reference
    approach direction (see generate_candidates) onto that sampled
    direction, with no free roll parameter - exactly the "sample directions,
    let IK filter the unreachable ones" scheme requested, since the sphere
    has no privileged roll axis to scan separately."""
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
        # a and b are (near-)antiparallel: any axis perpendicular to a works.
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis = axis / np.linalg.norm(axis)
        return Rot.from_rotvec(axis * math.pi).as_matrix()

    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def spherical_fibonacci_directions(n):
    """N roughly-uniformly-spaced unit vectors over the full sphere, via a
    Spherical Fibonacci lattice (same sampling scheme used for hand-eye
    calibration pose diversity)."""
    golden_ratio = (1.0 + 5.0 ** 0.5) / 2.0
    directions = []
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        r_xy = math.sqrt(max(0.0, 1.0 - z * z))
        theta = 2.0 * math.pi * i / golden_ratio
        directions.append(np.array([r_xy * math.cos(theta), r_xy * math.sin(theta), z]))
    return directions


def closest_point_on_ray_to_point(ray_origin, ray_dir, point):
    """Closest point on a ray (parameter clamped to >= 0) to a fixed point."""
    d = ray_dir / np.linalg.norm(ray_dir)
    t = np.dot(point - ray_origin, d)
    t = max(t, 0.0)
    return ray_origin + t * d


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class SphereGraspPlanner(object):
    def __init__(self):
        self.inertial_frame = rospy.get_param("~inertial_frame", INERTIAL_FRAME_DEFAULT)
        self.n_directions = rospy.get_param("~n_direction_samples", 30)
        self.axis_marker_length = rospy.get_param("~axis_marker_length", 0.02)
        self.tf_timeout = rospy.Duration(rospy.get_param("~tf_timeout", 1.0))

        self.min_cap_points = rospy.get_param("~min_cap_points", 50)
        self.cap_half_angle = math.radians(rospy.get_param("~cap_half_angle_deg", 40.0))

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.marker_pub = rospy.Publisher(
            "/grasp_planning/sphere_candidate_grasps", MarkerArray, queue_size=1, latch=True)
        self.best_marker_pub = rospy.Publisher(
            "/grasp_planning/best_sphere_grasp", MarkerArray, queue_size=1, latch=True)

        moveit_commander.roscpp_initialize(sys.argv)
        self.mgc = moveit_commander.MoveGroupCommander("right_arm")
        self.mgc.set_planning_time(0.5)

        rospy.wait_for_service('/compute_ik')
        self.ik_service = rospy.ServiceProxy('/compute_ik', GetPositionIK)

        self.grasp_service = rospy.Service(
            "/grasp_planning/get_sphere_grasp", GetSphereGraspPose, self.handle_get_grasp)
        rospy.loginfo("Sphere grasp planner ready on /grasp_planning/get_sphere_grasp")

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
    def build_sphere_estimate(self, req):
        """Packages the request's estimate+cloud into the working dict used
        throughout planning: pose in the inertial frame (for candidate
        generation) and pose/cloud in the cloud's native frame (for cap
        scoring - the cloud itself is never transformed, only small rays
        are). A sphere's orientation is not physically meaningful, but the
        estimate still carries one (identity, in practice) so the same
        pose-composition machinery as the cylinder/box planners applies
        unchanged."""
        pose_cloud_frame = req.estimate.pose  # PoseStamped, native perception frame

        try:
            pose_inertial = self.tf_buffer.transform(
                pose_cloud_frame, self.inertial_frame, timeout=self.tf_timeout)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Failed to transform sphere pose from %s to %s: %s",
                          pose_cloud_frame.header.frame_id, self.inertial_frame, e)
            return None

        inflation_margin = 0.0
        radius = req.estimate.diameter / 2.0
        return {
            "pose": pose_inertial,
            "pose_cloud_frame": pose_cloud_frame,
            "cloud": req.cloud,
            "radius": radius + inflation_margin,       # candidate generation - slightly inflated to improve robustness
            "radius_measured": radius,     # residual scoring; same value, no inflation needed here
        }

    # -- VMC-derived hand-relative sphere offset ----------------------------
    def compute_ball_vmc_offset(self, radius):
        """
        Direct port of the Julia VMC ball-placement snippet:
            ball_position = SVector(0.0, -ball_radius - 0.021, 0.33)
        This is the ball center's position in the forearm's own local frame
        at the reference (theta=0-equivalent) candidate orientation - i.e.
        forearm-to-ball, expressed in forearm-local coordinates. Unlike the
        cylinder, the clearance (0.021) is already folded into the formula
        as given, so no separate inflated-radius candidate-generation value
        is needed (see build_sphere_estimate).
        """
        return np.array([0.0, -radius - BALL_APPROACH_CLEARANCE, 0.33])

    # -- Candidate pose generation -------------------------------------------
    def generate_candidates(self, ball_pose_stamped, radius):
        """
        Generates candidate rh_palm poses for a SPHERE by sampling approach
        directions over a Spherical Fibonacci lattice. There is no
        analogue of the cylinder's axial scan or the box's width scan - a
        sphere has no privileged translational DOF beyond which direction
        the hand approaches from, so direction sampling is the only
        candidate-generation axis, exactly as agreed: sample broadly, let
        IK/planning reject what's unreachable.
        """
        p_hand_ref_point = self.compute_ball_vmc_offset(radius)
        ref_norm = np.linalg.norm(p_hand_ref_point)
        if ref_norm < 1e-9:
            rospy.logerr("Degenerate ball VMC offset (zero norm) - cannot derive an approach direction.")
            return []
        ref_direction = p_hand_ref_point / ref_norm

        directions = spherical_fibonacci_directions(self.n_directions)

        p_ball = np.array([ball_pose_stamped.pose.position.x,
                            ball_pose_stamped.pose.position.y,
                            ball_pose_stamped.pose.position.z])
        q_ball = ball_pose_stamped.pose.orientation
        R_ball = Rot.from_quat([q_ball.x, q_ball.y, q_ball.z, q_ball.w]).as_matrix()

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

        R0 = np.array([[-1, 0, 0],
                       [0, 0, -1],
                       [0, -1, 0]])

        candidates = []
        for d in directions:
            # Rotate the reference forearm orientation so that its fixed
            # forearm->ball offset now points along the sampled direction d
            # (in ball-local coordinates). No separate translation DOF: the
            # target is always the ball's own origin (its center).
            R_local = rotation_aligning_vectors(ref_direction, d)
            t_local = -R_local @ p_hand_ref_point

            R_forearm_world = R_ball @ R_local
            t_forearm_world = R_ball @ t_local + p_ball

            R_palm_world = R_forearm_world @ R_forearm_to_palm @ R0
            t_palm_world = R_forearm_world @ t_forearm_to_palm + t_forearm_world

            pose = Pose()
            pose.position = Point(*t_palm_world)
            q_palm = Rot.from_matrix(R_palm_world).as_quat()
            pose.orientation = Quaternion(*q_palm)
            candidates.append(pose)

        return candidates

    # -- Visualization -----------------------------------------------------
    def make_axis_marker(self, pose, axis_index, marker_id, frame_id, stamp, ns="sphere_grasp_candidates"):
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
                       "/grasp_planning/sphere_candidate_grasps",
                       len(candidates), len(marker_array.markers))

    def publish_best_candidate(self, palm_pose, frame_id):
        marker_array = MarkerArray()
        stamp = rospy.Time.now()
        for axis_index in range(3):
            marker_array.markers.append(
                self.make_axis_marker(palm_pose, axis_index, axis_index, frame_id, stamp,
                                       ns="best_sphere_grasp"))
        self.best_marker_pub.publish(marker_array)

    # -- IK / planning filtering ---------------------------------------------
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
        ranking (see select_best_grasp)."""
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
        implied by that same approach_flange target - not an independent
        palm-frame offset), mirroring the cylinder/box planners."""
        approach_distance = 0.12  # m
        local_offset = np.array([0.0, 0.0, -approach_distance])

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

    # -- Cap scoring / ranking (point-count + residual, cylinder-style) -----
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

    def score_candidate_cap(self, ball_estimate, ray_origin_world, ray_dir_world):
        """Cylinder's sector-residual idea, adapted to a sphere: instead of
        an axial band + angular sector around a point on a cylinder's
        surface, this scores an angular CAP (great-circle radius
        cap_half_angle) around the point on the sphere the candidate's
        approach ray is heading toward. No axial band is needed - a sphere
        has no axis."""
        cloud_frame = ball_estimate["pose_cloud_frame"].header.frame_id

        ray_origin_cloud, ray_dir_cloud = self.transform_ray_to_frame(
            ray_origin_world, ray_dir_world, self.inertial_frame, cloud_frame)

        ball_pose = ball_estimate["pose_cloud_frame"].pose
        center = np.array([ball_pose.position.x, ball_pose.position.y, ball_pose.position.z])

        p_ray = closest_point_on_ray_to_point(ray_origin_cloud, ray_dir_cloud, center)
        contact_dir = p_ray - center
        contact_norm = np.linalg.norm(contact_dir)
        if contact_norm < 1e-9:
            # Ray passes essentially through the center - fall back to
            # "coming from behind the ray origin" as the cap axis.
            contact_dir = -ray_dir_cloud / np.linalg.norm(ray_dir_cloud)
        else:
            contact_dir = contact_dir / contact_norm

        points = self.cloud_to_array(ball_estimate["cloud"])
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

        residual = float(np.sqrt(np.mean((r[mask] - ball_estimate["radius_measured"]) ** 2)))
        return n_points, residual

    def select_best_grasp(self, candidates, ball_estimate):
        scored = []
        for cand in candidates:
            palm = cand["palm"]
            origin = np.array([palm.position.x, palm.position.y, palm.position.z])
            q = palm.orientation
            R_palm = Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            direction = R_palm[:, 2]  # palm local approach direction

            n_points, residual = self.score_candidate_cap(ball_estimate, origin, direction)
            scored.append({"candidate": cand, "n_points": n_points, "residual": residual})

        qualifying = [s for s in scored if s["n_points"] >= self.min_cap_points]

        if qualifying:
            best = min(qualifying, key=lambda s: s["residual"])
            best["met_confidence_threshold"] = True
            rospy.loginfo("Selected grasp by cap residual fit (%d/%d candidates had >= %d cap points).",
                           len(qualifying), len(scored), self.min_cap_points)
        else:
            best = max(scored, key=lambda s: s["n_points"])
            best["met_confidence_threshold"] = False
            rospy.logwarn("No candidate reached %d cap points - falling back to "
                          "the candidate with the most cap evidence (%d points).",
                          self.min_cap_points, best["n_points"])

        return best["candidate"], best

    # -- Service handler -----------------------------------------------------
    def handle_get_grasp(self, req):
        res = GetSphereGraspPoseResponse()

        if req.estimate.primitive_type != "SPHERE":
            res.success = False
            res.reason = "Request primitive_type is not SPHERE."
            return res

        ball_estimate = self.build_sphere_estimate(req)
        if ball_estimate is None:
            res.success = False
            res.reason = "Failed to transform sphere pose into the inertial frame."
            return res

        candidates = self.generate_candidates(ball_estimate["pose"], ball_estimate["radius"])
        if not candidates:
            res.success = False
            res.reason = "No candidate grasp poses were generated."
            return res

        self.publish_candidates(candidates, self.inertial_frame)

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

        #self.publish_candidates([c["approach_flange"] for c in valid_candidates], self.inertial_frame)

        best_candidate, score_info = self.select_best_grasp(valid_candidates, ball_estimate)
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
    rospy.init_node("sphere_grasp_planner_node")
    SphereGraspPlanner()
    rospy.spin()


if __name__ == "__main__":
    main()