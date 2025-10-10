#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64, Int32
import numpy as np
import time

class HandPreShape:
    def __init__(self):
        rospy.init_node('hand_preshape', anonymous=True)

        self.mode_names = ["reset", "medium_wrap", "power_sphere", "lateral_pinch"]

        self.pub_topics = [
            "/sh_rh_wrj1_position_controller/command", "/sh_rh_wrj2_position_controller/command",
            "/sh_rh_ffj0_position_controller/command", "/sh_rh_ffj3_position_controller/command", "/sh_rh_ffj4_position_controller/command",
            "/sh_rh_mfj0_position_controller/command", "/sh_rh_mfj3_position_controller/command", "/sh_rh_mfj4_position_controller/command",
            "/sh_rh_rfj0_position_controller/command", "/sh_rh_rfj3_position_controller/command",
            "/sh_rh_rfj4_position_controller/command", "/sh_rh_lfj0_position_controller/command",
            "/sh_rh_lfj3_position_controller/command", "/sh_rh_lfj4_position_controller/command",  
            "/sh_rh_lfj5_position_controller/command", "/sh_rh_thj1_position_controller/command", "/sh_rh_thj2_position_controller/command", 
            "/sh_rh_thj3_position_controller/command", "/sh_rh_thj4_position_controller/command", "/sh_rh_thj5_position_controller/command"
        ]

        self.publishers = [rospy.Publisher(topic, Float64, queue_size=10) for topic in self.pub_topics]
        self.rate = rospy.Rate(10)  # 10 Hz

        # Subscribe to /preshape topic
        self.sub = rospy.Subscriber("/preshape", Int32, self.preshape_callback)

        rospy.loginfo("Hand preshape node initialized. Waiting for preshape commands on /preshape (0, 1, 2, or 3)...")

    def get_joint_positions(self, mode):
        if mode == 0:  # Reset hand
            positions = np.zeros(len(self.pub_topics))
        elif mode == 1:  # Medium Wrap
            positions = np.zeros(len(self.pub_topics))
            positions[18] = 1.2  # THJ4
        elif mode == 2:  # Power Sphere
            positions = [0.0, 0.0, 0.0, 0.0, -0.35, 
                         0.0, 0.0, -0.12, 0.0, 0.0, -0.12, 0.0, 0.0, -0.35, 0.0,
                         0.0, 0.0, 0.0, 1.22, 0.0]
        elif mode == 3:  # Lateral Pinch
            positions = [0.0, 0.0, 1.57, 1.0, 0.0, 
                         3.14, 1.57, 0.0, 3.14, 1.57, 0.0, 3.14, 1.57, 0.0, 0.0, 
                         0.0, -0.7, 0.0, 0.0, 0.0]
        else:
            rospy.logwarn(f"Unknown mode {mode}, defaulting to reset.")
            positions = np.zeros(len(self.pub_topics))
        return positions

    def publish_for_duration(self, positions, duration=1.0):
        """Publish the given joint positions at 10 Hz for `duration` seconds."""
        start_time = time.time()
        while not rospy.is_shutdown() and (time.time() - start_time < duration):
            for pub, val in zip(self.publishers, positions):
                pub.publish(Float64(val))
            self.rate.sleep()

    def preshape_callback(self, msg):
        mode = msg.data
        if mode not in (0, 1, 2, 3):
            rospy.logwarn(f"Ignoring invalid preshape mode: {mode}")
            return

        rospy.loginfo(f"Publishing preshape {self.mode_names[mode]} for 1 second")
        positions = self.get_joint_positions(mode)
        self.publish_for_duration(positions, duration=1.0)

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        node = HandPreShape()
        node.run()
    except rospy.ROSInterruptException:
        pass
