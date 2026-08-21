#!/usr/bin/env python3
"""
flatbox_grasp_planner_node.py

Service node: given a stable FLAT_BOX primitive estimate + its consensus
point cloud (as produced by /perception/get_stable_estimate and forwarded by
the grasp-planning dispatcher, exactly as for the cylinder planner), generates
candidate rh_palm grasp poses for a lateral pinch grasp on the box's near
face, filters them through IK/motion-planning feasibility, and returns a
selected candidate.

NOTE on selection: unlike the cylinder planner, there is no point-cloud-based
fit-quality score here (see select_best_grasp docstring) - the box is not
rotationally symmetric and a single-camera cloud only ever sees the one face
being grasped, so the cylinder's sector-residual approach doesn't carry over
without further work. Candidates are currently ranked by how close their
width offset is to the box's centerline, as a reasonable placeholder.
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

from my_package.srv import GetFlatBoxGraspPose, GetFlatBoxGraspPoseResponse

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

LATERAL_PINCH_PRESHAPE_WIDTH = 0.085  # m

# Fixed rotation mapping HAND-frame axes to FLAT_BOX-local-frame axes, at the
# theta=0 reference orientation: maps hand's +X onto flat_box's +Z
R0 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class FlatBoxGraspPlanner(object):
    def __init__(self):
        self.inertial_frame = rospy.get_param("~inertial_frame", INERTIAL_FRAME_DEFAULT)
        self.n_box_width = rospy.get_param("~n_box_width_samples", 5)
        self.axis_marker_length = rospy.get_param("~axis_marker_length", 0.02)
        self.tf_timeout = rospy.Duration(rospy.get_param("~tf_timeout", 1.0))

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.marker_pub = rospy.Publisher(
            "/grasp_planning/flatbox_candidate_grasps", MarkerArray, queue_size=1, latch=True)
        self.best_marker_pub = rospy.Publisher(
            "/grasp_planning/best_flatbox_grasp", MarkerArray, queue_size=1, latch=True)

        moveit_commander.roscpp_initialize(sys.argv)
        self.mgc = moveit_commander.MoveGroupCommander("right_arm")
        self.mgc.set_planning_time(0.5)  # default is usually 5.0s

        rospy.wait_for_service('/compute_ik')
        self.ik_service = rospy.ServiceProxy('/compute_ik', GetPositionIK)

        self.grasp_service = rospy.Service(
            "/grasp_planning/get_flatbox_grasp", GetFlatBoxGraspPose, self.handle_get_grasp)
        rospy.loginfo("Flat box grasp planner ready on /grasp_planning/get_flatbox_grasp")

    # -- Estimate handling (from request, not a service call) --------------
    def build_flatbox_estimate(self, req):
        """Packages the request's estimate+cloud into the working dict used
        throughout planning: pose in the inertial frame (for candidate
        generation) and pose/cloud in the cloud's native frame (kept around
        for parity with the cylinder planner and any future scoring work -
        the cloud itself is never transformed)."""
        pose_cloud_frame = req.estimate.pose  # PoseStamped, native perception frame

        try:
            pose_inertial = self.tf_buffer.transform(
                pose_cloud_frame, self.inertial_frame, timeout=self.tf_timeout)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Failed to transform flat box pose from %s to %s: %s",
                          pose_cloud_frame.header.frame_id, self.inertial_frame, e)
            return None

        return {
            "pose": pose_inertial,
            "pose_cloud_frame": pose_cloud_frame,
            "cloud": req.cloud,
            "thickness": req.estimate.thickness,
            "width": req.estimate.width,
            "depth": req.estimate.depth,
        }

    def compute_box_vmc_offset(self, thickness, depth):
        """
        Direct port of the Julia VMC box-placement snippet:
            box_dimensions = [box_thickness, box_width, 0.1]
            box_position = SVector(0.042 + box_dimensions[1], -0.03, 0.32 + box_dimensions[3])
        """
        return np.array([0.042 + thickness / 2, -0.03, 0.32 + depth / 2])

    def generate_box_candidates(self, box_pose_stamped, width, thickness, depth):
        """
        Generates candidate rh_palm poses for a FLAT_BOX, targeting the face
        perpendicular to the depth axis with the LOWER depth coordinate (the
        face closest to the camera - see box_pose_stamped's orientation
        convention: local X = width_dir, local Y = depth_dir, local Z = normal).

        Returns a list of {"palm": Pose, "width_offset": float} dicts. The
        width_offset is carried through the whole pipeline (IK/planning
        filtering preserve unknown dict keys) so select_best_grasp can rank
        surviving candidates by how centered they are on the box, in place of
        the point-cloud-based score the cylinder planner uses.
        """
        offset = self.compute_box_vmc_offset(thickness, depth)
        p_hand_ref_point = offset  # already the full [x, y, z], unlike the cylinder's [0, y, z]

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

        # Box pose in inertial frame
        p_box = np.array([box_pose_stamped.pose.position.x,
                           box_pose_stamped.pose.position.y,
                           box_pose_stamped.pose.position.z])
        q_box = box_pose_stamped.pose.orientation
        R_box = Rot.from_quat([q_box.x, q_box.y, q_box.z, q_box.w]).as_matrix()

        # Current rh_forearm -> rh_palm transform
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

        R_local = R0  # single fixed orientation - the box is not rotationally symmetric, so no theta scan

        candidates = []
        for w in width_positions:
            # Target: a point on the near (low-depth, camera-facing) face,
            # offset by w along the box's own width axis (local X). The
            # "-depth/2" is already handled by compute_box_vmc_offset.
            target_box_local = np.array([w, 0.0, 0.0])

            # Solve for rh_forearm origin (in box-local frame) such that the
            # known hand-frame reference point maps onto that target.
            t_local = target_box_local - R_local @ p_hand_ref_point

            # Compose rh_forearm candidate pose into the inertial frame
            R_forearm_world = R_box @ R_local
            t_forearm_world = R_box @ t_local + p_box

            # Compose with the current forearm->palm transform for rh_palm
            R_palm_world = R_forearm_world @ R_forearm_to_palm
            t_palm_world = R_forearm_world @ t_forearm_to_palm + t_forearm_world

            pose = Pose()
            pose.position = Point(*t_palm_world)
            q_palm = Rot.from_matrix(R_palm_world).as_quat()
            pose.orientation = Quaternion(*q_palm)
            candidates.append({"palm": pose, "width_offset": float(w)})

        return candidates

    # -- Visualization -----------------------------------------------------
    def make_axis_marker(self, pose, axis_index, marker_id, frame_id, stamp, ns="flatbox_grasp_candidates"):
        """
        Builds one ARROW marker representing one local axis (0=X/red, 1=Y/green,
        2=Z/blue) of the given candidate pose. ARROW markers point along their
        own local +X by default, so Y/Z axes are represented by additionally
        rotating the marker's orientation.
        """
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
                       "/grasp_planning/flatbox_candidate_grasps",
                       len(candidates), len(marker_array.markers))

    def publish_best_candidate(self, palm_pose, frame_id):
        marker_array = MarkerArray()
        stamp = rospy.Time.now()
        for axis_index in range(3):
            marker_array.markers.append(
                self.make_axis_marker(palm_pose, axis_index, axis_index, frame_id, stamp,
                                       ns="best_flatbox_grasp"))
        self.best_marker_pub.publish(marker_array)

    # -- IK / planning filtering (candidate = dict; unknown keys preserved) -
    def filter_grasps_with_ik(self, candidates, pose_key):
        """Filters a list of candidate dicts, testing IK on candidate[pose_key].
        Candidates (with ALL their keys, e.g. 'palm', 'width_offset') are
        preserved unmodified for those that pass."""
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

    def transform_candidates_palm_to_flange(self, palm_candidates):
        """Pairs each rh_palm candidate dict with its ra_flange target,
        preserving any extra keys already on the candidate (e.g.
        'width_offset'). Returns [{'palm': Pose, 'flange': Pose, ...}, ...]."""
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

    def compute_approach_poses(self, candidates):
        """Adds 'approach_flange' (offset backward along the flange's local X,
        a MoveIt/IK convenience standoff) and 'approach_palm' (the palm pose
        implied by that same approach_flange target - not an independent
        palm-frame offset), mirroring the cylinder planner. All other
        candidate keys (e.g. 'width_offset') are preserved."""
        approach_distance_axial = 0.12  # m
        approach_distance_orthogonal = 0.05  # m
        local_offset = np.array([-approach_distance_axial, approach_distance_orthogonal, 0.0])

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

    # -- Selection ------------------------------------------------------------
    def select_best_grasp(self, candidates):
        """Picks the surviving candidate whose width_offset is closest to the
        box's centerline (0.0).

        This is a placeholder ranking, not a fit-quality score: the
        cylinder planner's sector-residual metric works because a partial
        cloud can still confirm a radius all the way around the predicted
        contact band. A flat face has no equivalent rotational invariant,
        and a single-camera cloud only ever observes the one face being
        grasped, so there's no independent geometric check left to score
        candidates against. Centering is used instead as the one property
        that plausibly correlates with contact quality without additional
        information (e.g. estimated edge locations).
        """
        return min(candidates, key=lambda c: abs(c["width_offset"]))

    # -- Service handler -----------------------------------------------------
    def handle_get_grasp(self, req):
        res = GetFlatBoxGraspPoseResponse()

        if req.estimate.primitive_type != "FLAT_BOX":
            res.success = False
            res.reason = "Request primitive_type is not FLAT_BOX."
            return res

        estimate = self.build_flatbox_estimate(req)
        if estimate is None:
            res.success = False
            res.reason = "Failed to transform flat box pose into the inertial frame."
            return res

        candidates = self.generate_box_candidates(
            estimate["pose"], estimate["width"], estimate["thickness"], estimate["depth"])
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

        best_candidate = self.select_best_grasp(valid_candidates)
        self.publish_best_candidate(best_candidate["palm"], self.inertial_frame)

        res.success = True
        res.grasp_pose_palm = best_candidate["palm"]
        res.grasp_pose_flange = best_candidate["flange"]
        res.approach_pose_palm = best_candidate["approach_palm"]
        res.approach_pose_flange = best_candidate["approach_flange"]
        res.candidate_count = len(valid_candidates)
        res.selected_width_offset = best_candidate["width_offset"]

        return res


def main():
    rospy.init_node("flatbox_grasp_planner_node")
    FlatBoxGraspPlanner()
    rospy.spin()


if __name__ == "__main__":
    main()