#!/usr/bin/env python3
"""
ROS Node: Joint Command Filter
-------------------------------
Subscribes to:
  - /shadowhand_command_filtering  (std_msgs/Float64MultiArray) — desired joint commands
  - /shadowhand_state_topic        (std_msgs/Float64MultiArray) — current joint states

Publishes to:
  - /shadowhand_command_topic      (std_msgs/Float64MultiArray) — filtered (interpolated) commands

Behaviour:
  When a new command is received, the node interpolates from the current
  joint state to the target command over INTERPOLATION_DURATION seconds,
  publishing at PUBLISH_RATE Hz.  If a new command arrives mid-interpolation,
  the current interpolated position becomes the new start point and the
  interpolation restarts toward the new target.
"""

import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
from typing import Optional
import threading


# ── Configuration ──────────────────────────────────────────────────────────────
INTERPOLATION_DURATION = 1.0   # seconds to reach the target command
PUBLISH_RATE           = 100   # Hz — interpolation / publish frequency
# ───────────────────────────────────────────────────────────────────────────────


class JointCommandFilter:
    def __init__(self):
        rospy.init_node("joint_command_filter", anonymous=False)

        self._lock = threading.Lock()

        # Joint state (current actual or last known interpolated position)
        self._current_positions = None  # type: Optional[np.ndarray]

        # Interpolation bookkeeping
        self._interp_start   = None   # type: Optional[np.ndarray]  # positions when target was set
        self._interp_target  = None   # type: Optional[np.ndarray]  # desired target positions
        self._interp_start_t = 0.0    # type: float                 # ROS time when interpolation began
        self._interpolating  = False  # type: bool

        # ── Subscribers ────────────────────────────────────────────────────────
        rospy.Subscriber(
            "/shadowhand_command_filtering",
            Float64MultiArray,
            self._command_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "/shadowhand_state_topic",
            Float64MultiArray,
            self._state_callback,
            queue_size=1,
        )

        # ── Publisher ──────────────────────────────────────────────────────────
        self._pub = rospy.Publisher(
            "/shadowhand_command_topic",
            Float64MultiArray,
            queue_size=1,
        )

        rospy.loginfo("[JointCommandFilter] Node started.")
        rospy.loginfo(
            f"[JointCommandFilter] Interpolation duration: {INTERPOLATION_DURATION}s  |  "
            f"Publish rate: {PUBLISH_RATE} Hz"
        )

    # ── Callbacks ───────────────────────────────────────────────────────────────

    def _state_callback(self, msg):
        """Keep track of the real joint state (used only for the very first
        command when we have no interpolated position yet)."""
        with self._lock:
            state = np.array(msg.data, dtype=float)
            if self._current_positions is None:
                # Initialise from hardware state on first message
                self._current_positions = state.copy()
                rospy.loginfo(
                    f"[JointCommandFilter] Initialised {len(state)} joint(s) from state topic."
                )

    def _command_callback(self, msg):
        """Receive a new target command and (re)start interpolation."""
        target = np.array(msg.data, dtype=float)

        with self._lock:
            # If we don't know where we are yet, hold until state arrives
            if self._current_positions is None:
                rospy.logwarn_once(
                    "[JointCommandFilter] Command received before joint state — "
                    "waiting for /shadowhand_state_topic."
                )
                return

            # Handle dimension mismatch gracefully
            if len(target) != len(self._current_positions):
                rospy.logwarn(
                    f"[JointCommandFilter] Command has {len(target)} joints but "
                    f"state has {len(self._current_positions)} — ignoring."
                )
                return

            # Start interpolation from wherever we currently are
            self._interp_start   = self._current_positions.copy()
            self._interp_target  = target.copy()
            self._interp_start_t = rospy.get_time()
            self._interpolating  = True

            rospy.logdebug("[JointCommandFilter] New target received; interpolation (re)started.")

    # ── Main loop ───────────────────────────────────────────────────────────────

    def _compute_interpolated(self):
        """Return the interpolated position for the current time (CALL WITH LOCK)."""
        if not self._interpolating:
            return self._current_positions  # nothing to do

        elapsed = rospy.get_time() - self._interp_start_t
        alpha   = min(elapsed / INTERPOLATION_DURATION, 1.0)

        # Linear interpolation (LERP): start + α * (target - start)
        interpolated = self._interp_start + alpha * (self._interp_target - self._interp_start)

        if alpha >= 1.0:
            self._interpolating       = False
            self._current_positions   = self._interp_target.copy()
            rospy.logdebug("[JointCommandFilter] Target reached.")
        else:
            self._current_positions = interpolated.copy()

        return interpolated

    def run(self):
        rate = rospy.Rate(PUBLISH_RATE)

        while not rospy.is_shutdown():
            with self._lock:
                output = self._compute_interpolated()

            if output is not None:
                out_msg      = Float64MultiArray()
                out_msg.data = output.tolist()
                self._pub.publish(out_msg)

            rate.sleep()


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        node = JointCommandFilter()
        node.run()
    except rospy.ROSInterruptException:
        pass