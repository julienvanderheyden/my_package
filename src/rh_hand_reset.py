#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64
import time 

class HandReset:
    def __init__(self):
        rospy.init_node('hand_reset', anonymous=True)
        
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
        
    def publish_zero_commands(self):
        rospy.loginfo("Waiting for 10 seconds before publishing zero commands...")
        time.sleep(10)  # Wait for 10 seconds before sending the reset commands
        
        rospy.loginfo("Publishing zero commands to all hand joints...")
        zero_msg = Float64(0.0)
        while not rospy.is_shutdown():
            for pub in self.publishers:
                pub.publish(zero_msg)
            self.rate.sleep()

if __name__ == "__main__":
    try:
        node = HandReset()
        node.publish_zero_commands()
    except rospy.ROSInterruptException:
        pass
