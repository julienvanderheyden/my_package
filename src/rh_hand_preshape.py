#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64
import numpy as np
import time
import sys

class HandPreShape:
    def __init__(self, mode):
        rospy.init_node('hand_preshape', anonymous=True)

        self.mode = mode
        self.mode_names = [ "reset", "medium_wrap", "power_sphere", "lateral_pinch"]
        
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

        self.positions = self.get_joint_positions(mode)

        self.publishers = [rospy.Publisher(topic, Float64, queue_size=10) for topic in self.pub_topics]
        self.rate = rospy.Rate(10)  # 10 Hz

    def get_joint_positions(self, mode):
        if mode == 0 : 
            # Reset hand
            positions = np.zeros(len(self.pub_topics))
            return positions 
        if mode == 1:
            # Medium Wrap
            positions = np.zeros(len(self.pub_topics))
            positions[18] = 1.2 #THJ4
            return positions
        elif mode == 2:
            # POWER SPHERE : TODO
            positions = [0.2 for _ in self.pub_topics]
            return positions
        elif mode == 3:
            # lateral pinch
            positions = [0.0, 0.0, 1.57, 1.0, 0.0, 
                         3.14, 1.57, 0.0, 3.14, 1.57, 0.0, 3.14, 1.57, 0.0, 0.0, 
                         0.0, -0.7, 0.0, 0.0, 0.0] 
            return positions
        else:
            rospy.logwarn(f"Unknown mode: {mode}. Defaulting to zeros.")
            return [0.0 for _ in self.pub_topics]

    def publish_position_commands(self):
        rospy.loginfo(f"Waiting for 5 seconds before publishing commands...")
        time.sleep(5)

        rospy.loginfo(f"Publishing joint commands for mode {self.mode_names[self.mode]}...")
        while not rospy.is_shutdown():
            for pub, val in zip(self.publishers, self.positions):
                pub.publish(Float64(val))
            self.rate.sleep()

if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print("Usage: rosrun <package> hand_reset.py <mode: 0|1|2|3>")
            sys.exit(1)

        mode = int(sys.argv[1])
        if mode not in (0, 1, 2, 3):
            print("Invalid mode. Must be 0, 1, 2, or 3.")
            sys.exit(1)

        node = HandPreShape(mode)
        node.publish_position_commands()

    except rospy.ROSInterruptException:
        pass
