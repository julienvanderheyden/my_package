#!/usr/bin/env python
"""
Diagnostic node: isolates whether a depth<->color alignment error exists
INSIDE THE CAMERA ITSELF, with no robot, TF, or hand-eye calibration
involved at any point.

Method:
  1. Detect the ChArUco board in the RGB image. For each detected
     interior corner, we get BOTH:
       (a) its pixel coordinate (u, v)
       (b) a 3D position in camera_color_optical_frame, computed purely
           from solvePnP (board geometry + intrinsics + detected 2D
           corners). This does NOT use depth data at all.
  2. For that same pixel (u, v), read the ALIGNED depth image and
     deproject it into a 3D point using the color camera intrinsics.
     This gives a second, independent 3D position for the same physical
     corner, this time derived entirely from depth data.
  3. Compare (a) and (b) for every corner. A systematic offset here
     can only come from the depth<->color alignment inside the camera
     / SDK, since neither measurement touches the robot, TF tree, or
     hand-eye calibration.

Subscribes:
    /camera/color/image_raw            (or image_rect_color)
    /camera/color/camera_info
    /camera/aligned_depth_to_color/image_raw

Prints, for each synced frame: per-corner offset vectors (PnP - depth),
and the mean offset across all detected corners (in meters, in the
camera_color_optical_frame convention: +X right, +Y down, +Z forward).

Also publishes a MarkerArray on /charuco_alignment_check/markers with
both sets of corner points (PnP in one color, depth-derived in
another) for direct visual comparison in RViz.
"""
import rospy
import cv2
import numpy as np
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class CharucoDepthAlignmentCheck:
    def __init__(self):
        # --- Board parameters: defaults match the board configured in
        # camera_calibration.launch (DICT_5X5_50, 5x5 squares, 20mm
        # square / 14mm marker -> 100mm board). Override via rosparam
        # if yours differs.
        dict_name = rospy.get_param("~dictionary", "DICT_5X5_50")
        squares_x = rospy.get_param("~squares_x", 5)
        squares_y = rospy.get_param("~squares_y", 5)
        square_size = rospy.get_param("~square_size", 0.02)   # meters
        marker_size = rospy.get_param("~marker_size", 0.014)  # meters

        aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
        self.board = cv2.aruco.CharucoBoard_create(
            squares_x, squares_y, square_size, marker_size, aruco_dict)
        self.aruco_dict = aruco_dict

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.frame_id = None

        self.pub_markers = rospy.Publisher(
            "/charuco_alignment_check/markers", MarkerArray, queue_size=1)

        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_cb)

        image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
        ts = message_filters.ApproximateTimeSynchronizer(
            [image_sub, depth_sub], queue_size=5, slop=0.05)
        ts.registerCallback(self.synced_cb)

        rospy.loginfo("charuco_depth_alignment_check started, waiting for camera_info + frames...")

    def camera_info_cb(self, msg):
        self.camera_matrix = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.D, dtype=np.float64)
        self.frame_id = msg.header.frame_id

    def synced_cb(self, image_msg, depth_msg):
        if self.camera_matrix is None:
            return

        rgb = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        # aligned depth is typically 16UC1 in millimeters
        depth_scale = 0.001 if depth.dtype == np.uint16 else 1.0

        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict)
        if ids is None or len(ids) == 0:
            rospy.logwarn_throttle(5.0, "No ArUco markers detected in this frame")
            return

        n_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, self.board)
        if n_corners < 4:
            rospy.logwarn_throttle(5.0, "Not enough ChArUco corners interpolated (%d)" % n_corners)
            return

        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, self.board,
            self.camera_matrix, self.dist_coeffs, None, None)
        if not ok:
            rospy.logwarn_throttle(5.0, "estimatePoseCharucoBoard failed")
            return

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)

        # 3D positions of each interior corner in the board's own frame
        board_corner_positions = self.board.chessboardCorners  # (N, 3)

        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]

        offsets = []
        pnp_points, depth_points = [], []

        for i, cid in enumerate(charuco_ids.flatten()):
            u, v = charuco_corners[i, 0]
            u_i, v_i = int(round(u)), int(round(v))

            # --- (a) PnP-derived 3D position, purely from RGB + board geometry ---
            p_board = board_corner_positions[cid]
            p_pnp = R.dot(p_board) + t  # in camera_color_optical_frame

            # --- (b) Depth-derived 3D position, at the SAME pixel ---
            if not (0 <= v_i < depth.shape[0] and 0 <= u_i < depth.shape[1]):
                continue
            z = float(depth[v_i, u_i]) * depth_scale
            if z <= 0.0:
                continue  # invalid/missing depth at this pixel
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            p_depth = np.array([x, y, z])

            offsets.append(p_pnp - p_depth)
            pnp_points.append(p_pnp)
            depth_points.append(p_depth)

        if not offsets:
            rospy.logwarn_throttle(5.0, "No corners had valid depth to compare against")
            return

        offsets = np.array(offsets)
        mean_offset = offsets.mean(axis=0)
        std_offset = offsets.std(axis=0)

        rospy.loginfo(
            "[%d corners] mean offset (PnP - depth) = [%.4f, %.4f, %.4f] m "
            "(std [%.4f, %.4f, %.4f]), norm=%.4f m",
            len(offsets), mean_offset[0], mean_offset[1], mean_offset[2],
            std_offset[0], std_offset[1], std_offset[2],
            float(np.linalg.norm(mean_offset)))

        self._publish_markers(pnp_points, depth_points, image_msg.header.frame_id)

    def _publish_markers(self, pnp_points, depth_points, frame_id):
        marker_array = MarkerArray()

        for ns, points, color in [
                ("pnp_corners", pnp_points, (0.0, 1.0, 0.0, 1.0)),      # green
                ("depth_corners", depth_points, (1.0, 0.0, 0.0, 1.0))]:  # red
            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = rospy.Time.now()
            m.ns = ns
            m.id = 0
            m.type = Marker.SPHERE_LIST
            m.action = Marker.ADD
            m.scale.x = m.scale.y = m.scale.z = 0.005
            m.color.r, m.color.g, m.color.b, m.color.a = color
            m.pose.orientation.w = 1.0
            m.points = [Point(x=p[0], y=p[1], z=p[2]) for p in points]
            marker_array.markers.append(m)

        self.pub_markers.publish(marker_array)


if __name__ == "__main__":
    rospy.init_node("charuco_depth_alignment_check")
    CharucoDepthAlignmentCheck()
    rospy.spin()