#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray

class DynamicRouter:
    def __init__(self):
        rospy.init_node('dynamic_router', anonymous=True)

        self.sub_topic = "/shadowhand_command_topic"
        self.pub_topics = ["/sh_rh_wrj1_position_controller/command", "/sh_rh_wrj2_position_controller/command", "/sh_rh_ffj0_position_controller/command",
                           "fake_topic", "/sh_rh_ffj3_position_controller/command", "/sh_rh_ffj4_position_controller/command", 
                           "/sh_rh_mfj0_position_controller/command", "/fake_topic", "/sh_rh_mfj3_position_controller/command", "/sh_rh_mfj4_position_controller/command", 
                           "/sh_rh_rfj0_position_controller/command", "/fake_topic", "/sh_rh_rfj3_position_controller/command",
                           "/sh_rh_rfj4_position_controller/command", "/sh_rh_lfj0_position_controller/command", "/fake_topic", 
                           "/sh_rh_lfj3_position_controller/command", "/sh_rh_lfj4_position_controller/command",  
                           "/sh_rh_lfj5_position_controller/command", "/sh_rh_thj1_position_controller/command", "/sh_rh_thj2_position_controller/command", 
                           "/sh_rh_thj3_position_controller/command", "/sh_rh_thj4_position_controller/command", "/sh_rh_thj5_position_controller/command"]

        self.publishers = [rospy.Publisher(topic, Float64, queue_size=10) for topic in self.pub_topics]
        rospy.Subscriber(self.sub_topic, Float64MultiArray, self.callback)

        rospy.loginfo(f"Subscribed to: {self.sub_topic}")
        rospy.loginfo(f"Publishing to: {self.pub_topics}")

    def callback(self, msg):
        data = msg.data  # Extract array from Float64MultiArray

        # Ensure the received data length matches the number of publishers
        if len(data) != len(self.publishers):
            rospy.logwarn(f"Received {len(data)} values, but expected {len(self.publishers)}. Ignoring message.")
            return

        # Publish each float to the corresponding topic
        for i in range(len(data)):
            self.publishers[i].publish(Float64(data[i]))

if __name__ == "__main__":
    try:
        node = DynamicRouter()
        rospy.spin()  # Keep the node running
    except rospy.ROSInterruptException:
        pass
