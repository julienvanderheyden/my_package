#!/usr/bin/env python3

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

LATERAL_PINCH_PRESHAPE_WIDTH = 0.085  # m


# Fixed rotation mapping HAND-frame axes to FLAT_BOX-local-frame axes, at the
# theta=0 reference orientation: maps hand's +X onto flat_box's +Z
R0 = Rot.from_euler('y', -90, degrees=True).as_matrix()


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class FlatBoxGraspPlanner(object):
    def __init__(self):
        self.inertial_frame = rospy.get_param("~inertial_frame", INERTIAL_FRAME_DEFAULT)
        self.n_box_width = rospy.get_param("~n_box_width_samples", 5)
        self.axis_marker_length = rospy.get_param("~axis_marker_length", 0.02)
        self.tf_timeout = rospy.Duration(rospy.get_param("~tf_timeout", 1.0))

        # -- Sector scoring / ranking params --
        self.min_sector_points = rospy.get_param("~min_sector_points", 15)
        self.sector_height_half_width = rospy.get_param(
            "~sector_height_half_width", LATERAL_PINCH_PRESHAPE_WIDTH / 2.0)  # m
        self.sector_angle_half_width = math.radians(
            rospy.get_param("~sector_angle_half_width_deg", 30.0))

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.marker_pub = rospy.Publisher(
            "/grasp_planning/flatbox_candidate_grasps", MarkerArray, queue_size=1, latch=True)
        self.best_marker_pub = rospy.Publisher(
            "/grasp_planning/best_flatbox_grasp", MarkerArray, queue_size=1, latch=True)

        rospy.wait_for_service("/perception/get_stable_estimate")
        self.perception_srv = rospy.ServiceProxy(
            "/perception/get_stable_estimate", GetStableEstimate)

        # moveit_commander.roscpp_initialize(sys.argv)
        # self.mgc = moveit_commander.MoveGroupCommander("right_arm")
        # self.mgc.set_planning_time(0.5)  # default is usually 5.0s

        # rospy.wait_for_service('/compute_ik')
        # self.ik_service = rospy.ServiceProxy('/compute_ik', GetPositionIK)

    # -- Perception -----------------------------------------------------
    def get_flatbox_estimate(self):
        """Returns the estimate of the flat box's pose and dimensions in the inertial frame."""
        try:
            resp = self.perception_srv()
        except rospy.ServiceException as e:
            rospy.logerr("Perception service call failed: %s", e)
            return None

        if not resp.success:
            rospy.logwarn("Perception service reports no stable estimate: %s", resp.reason)
            return None

        if resp.estimate.primitive_type != "FLAT_BOX":
            rospy.logwarn("Stable estimate is type '%s', not FLAT_BOX - nothing to plan for.",
                            resp.estimate.primitive_type)
            return None

        pose_cloud_frame = resp.estimate.pose  # PoseStamped, native frame (e.g. camera_color_optical_frame)
        
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
            "cloud": resp.cloud,
            "thickness": resp.estimate.thickness,  
            "width": resp.estimate.width,  
            "depth": resp.estimate.depth,
        }

    def compute_box_vmc_offset(self, thickness, depth):
        """
        Direct port of the Julia VMC box-placement snippet:
            box_dimensions = [box_thickness, box_width, 0.1]
            box_position = SVector(0.042 + box_dimensions[1], -0.03, 0.32 + box_dimensions[3])
        """
        return np.array([0.042 + thickness, -0.03, 0.32 + depth])

    def generate_box_candidates(self, box_pose_stamped, width, thickness, depth):
        """
        Generates candidate rh_palm poses for a FLAT_BOX, targeting the face
        perpendicular to the depth axis with the LOWER depth coordinate (the
        face closest to the camera - see box_pose_stamped's orientation
        convention: local X = width_dir, local Y = depth_dir, local Z = normal).

        box_pose_stamped.pose.position is the box's assumed full-depth center
        (fitBox places it at depth_low + depth/2 along depth_dir), not the
        near face - so the near face's center is computed explicitly as
        p_box - (depth/2)*depth_dir, and used as the target instead of the
        box origin. Diversity is added by sampling target points spread along
        the width axis rather than the single Julia candidate.

        Structurally this mirrors generate_candidates(): a fixed hand-frame
        reference point (compute_box_vmc_offset) is placed so it lands on a
        chosen target point (here: points along the near face, instead of
        points along the cylinder axis at height a). R0 is reused unchanged,
        under the same assumption as the cylinder case (hand's local +X maps
        to the wrap/pinch axis - here the box's normal/thickness axis - see
        module docstring point 1 and verify visually in RViz before trusting).
        """
        offset = self.compute_box_vmc_offset(thickness)
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

        R_local = R0  # single fixed orientation - the box is not rotationally symmetric, so no theta scan

        candidates = []
        for w in width_positions:
            # Target: a point on the near (low-depth, camera-facing) face,
            # offset by w along the box's own width axis (local X).
            target_box_local = np.array([w, -depth / 2.0, 0.0])

            # Solve for rh_forearm origin (in box-local frame) such that the
            # known hand-frame reference point maps onto that target.
            t_local = target_box_local - R_local @ p_hand_ref_point

            # Compose rh_forearm candidate pose into the inertial frame
            R_forearm_world = R_box @ R_local
            t_forearm_world = R_box @ t_local + p_box

            # Compose with the current forearm->palm transform to get the rh_palm candidate
            R_palm_world = R_forearm_world @ R_forearm_to_palm
            t_palm_world = R_forearm_world @ t_forearm_to_palm + t_forearm_world

            pose = Pose()
            pose.position = Point(*t_palm_world)
            q_palm = Rot.from_matrix(R_palm_world).as_quat()
            pose.orientation = Quaternion(*q_palm)
            candidates.append(pose)

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

    # -- Top-level entry point ---------------------------------------------
    def run_once(self):
        estimate = self.get_flatbox_estimate()
        if estimate is None:
            return False

        candidates = self.generate_box_candidates(
            estimate["pose"], estimate["width"], estimate["thickness"], estimate["depth"])
        if not candidates:
            rospy.logerr("No candidate grasp poses were generated.")
            return False

        self.publish_candidates(candidates, self.inertial_frame)

        # paired_candidates = self.transform_candidates_palm_to_flange(candidates)
        # if not paired_candidates:
        #     return False

        # valid_candidates = self.filter_grasps_with_ik(paired_candidates, pose_key="flange")
        # valid_candidates = self.compute_approach_poses(valid_candidates)
        # valid_candidates = self.filter_grasps_with_ik(valid_candidates, pose_key="approach_flange")
        # valid_candidates = self.filter_grasps_with_plans(valid_candidates)

        # if not valid_candidates:
        #     rospy.logerr("No candidate grasp pose survived IK/planning filtering.")
        #     return False

        # self.publish_candidates([c["approach_flange"] for c in valid_candidates], self.inertial_frame)
        # rospy.loginfo("Generated %d valid flat-box grasp candidates.", len(valid_candidates))

        # Candidate generation only for now - no ranking/selection step yet
        # (the cylinder's sector-based ranking doesn't carry over to a flat
        # face without adaptation, and wasn't part of this pass). Downstream
        # code should pick a candidate['approach_flange']/['flange'] pair
        # from valid_candidates itself until that's added.
        # self.valid_candidates = valid_candidates
        return True


def main():
    rospy.init_node("flatbox_grasp_planner_node")
    planner = FlatBoxGraspPlanner()

    success = planner.run_once()
    if not success:
        rospy.logwarn("Initial candidate generation failed - node will keep running; "
                       "call run_once() again (e.g. via a future service/topic trigger) "
                       "once perception/tf data is available.")

    rospy.spin()


if __name__ == "__main__":
    main()