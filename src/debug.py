#!/usr/bin/env python3
"""
Diagnostic node for hand-eye / point-cloud offset debugging.

Looks up the static transform between camera_color_optical_frame and
camera_depth_optical_frame, then republishes /camera/depth/color/points
TWICE: once with every point shifted by +baseline, once by -baseline.

IMPORTANT: the output clouds keep the SAME frame_id as the input
(camera_color_optical_frame). This is deliberate -- if we changed the
frame_id, RViz would just re-apply the correct TF and "undo" our
manual shift, which would defeat the point. By keeping the frame_id
fixed, the offset we add is a raw, unconditional shift of the point
coordinates themselves. This directly mimics what a wrong-signed or
wrong-magnitude depth<->color extrinsic would do inside the driver's
own align_depth reprojection.

Usage:
    rosrun <your_package> baseline_offset_diagnostic.py

Then in RViz, add all three clouds:
    /camera/depth/color/points               (original)
    /camera/depth/color/points_shift_plus     (+ baseline)
    /camera/depth/color/points_shift_minus    (- baseline)

Whichever one lines up best against known robot geometry / the ChArUco
board tells you whether a baseline sign/magnitude error in align_depth
is a plausible explanation for your offset -- and roughly what
fraction of the observed few-cm error it could account for (baseline
is typically ~15-25mm on the D435, so if your offset is much larger
than that, this alone can't be the full story).
"""
import rospy
import tf2_ros
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField


class BaselineOffsetDiagnostic:
    def __init__(self):
        self.color_frame = rospy.get_param("~color_frame", "camera_color_optical_frame")
        self.depth_frame = rospy.get_param("~depth_frame", "camera_depth_optical_frame")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pub_plus = rospy.Publisher(
            "/camera/depth/color/points_shift_plus", PointCloud2, queue_size=1)
        self.pub_minus = rospy.Publisher(
            "/camera/depth/color/points_shift_minus", PointCloud2, queue_size=1)

        self.offset = None  # cached (dx, dy, dz) in meters
        rospy.Timer(rospy.Duration(1.0), self.update_offset)

        self.sub = rospy.Subscriber(
            "/camera/depth/color/points", PointCloud2, self.cloud_cb, queue_size=1)

        rospy.loginfo("baseline_offset_diagnostic started: %s -> %s",
                       self.depth_frame, self.color_frame)

    def update_offset(self, _event):
        try:
            # translation of depth_frame expressed in color_frame
            # (i.e. how far the depth module sits from the color module)
            t = self.tf_buffer.lookup_transform(
                self.color_frame, self.depth_frame, rospy.Time(0), rospy.Duration(0.5))
            tr = t.transform.translation
            self.offset = np.array([tr.x, tr.y, tr.z], dtype=np.float32)
            rospy.loginfo_throttle(
                5.0,
                "depth->color baseline = [%.4f, %.4f, %.4f] m (norm=%.4f m)"
                % (tr.x, tr.y, tr.z, float(np.linalg.norm(self.offset))))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5.0, "TF lookup failed: %s" % str(e))

    def cloud_cb(self, msg):
        if self.offset is None:
            return  # haven't got the baseline yet, skip this frame

        dtype = self._dtype_from_fields(msg.fields, msg.point_step)
        try:
            cloud_arr = np.frombuffer(msg.data, dtype=dtype).copy()
        except ValueError as e:
            rospy.logwarn_throttle(5.0, "Failed to parse cloud: %s" % str(e))
            return

        for sign, pub in [(+1.0, self.pub_plus), (-1.0, self.pub_minus)]:
            shifted = cloud_arr.copy()
            shifted['x'] += sign * self.offset[0]
            shifted['y'] += sign * self.offset[1]
            shifted['z'] += sign * self.offset[2]

            out = PointCloud2()
            out.header = msg.header  # frame_id UNCHANGED on purpose, see module docstring
            out.height = msg.height
            out.width = msg.width
            out.fields = msg.fields
            out.is_bigendian = msg.is_bigendian
            out.point_step = msg.point_step
            out.row_step = msg.row_step
            out.is_dense = msg.is_dense
            out.data = shifted.tobytes()
            pub.publish(out)

    @staticmethod
    def _dtype_from_fields(fields, point_step):
        """Build a numpy structured dtype matching the PointCloud2 layout.
        Non-xyz fields (e.g. packed rgb) are carried through as opaque
        bytes -- only x/y/z get modified above."""
        type_map = {
            PointField.INT8: np.int8, PointField.UINT8: np.uint8,
            PointField.INT16: np.int16, PointField.UINT16: np.uint16,
            PointField.INT32: np.int32, PointField.UINT32: np.uint32,
            PointField.FLOAT32: np.float32, PointField.FLOAT64: np.float64,
        }
        names, formats, offsets = [], [], []
        for f in fields:
            if f.datatype not in type_map:
                continue
            names.append(f.name)
            formats.append(type_map[f.datatype])
            offsets.append(f.offset)
        return np.dtype({'names': names, 'formats': formats,
                          'offsets': offsets, 'itemsize': point_step})


if __name__ == "__main__":
    rospy.init_node("baseline_offset_diagnostic")
    BaselineOffsetDiagnostic()
    rospy.spin()