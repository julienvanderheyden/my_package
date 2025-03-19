#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray

class DynamicRouter:
    def __init__(self):
        rospy.init_node('dynamic_router', anonymous=True)

        self.sub_topic = "/shadowhand_command_topic"
        self.pub_topics = ["/my_robot/rh_WRJ1_position_controller/command", "/my_robot/rh_WRJ2_position_controller/command", "/my_robot/rh_FFJ1_position_controller/command",
                           "/my_robot/rh_FFJ2_position_controller/command", "/my_robot/rh_FFJ3_position_controller/command", "/my_robot/rh_FFJ4_position_controller/command", 
                           "/my_robot/rh_MFJ1_position_controller/command", "/my_robot/rh_MFJ2_position_controller/command", "/my_robot/rh_MFJ3_position_controller/command", 
                           "/my_robot/rh_MFJ4_position_controller/command", "/my_robot/rh_RFJ1_position_controller/command", "/my_robot/rh_RFJ2_position_controller/command",
                           "/my_robot/rh_RFJ3_position_controller/command", "/my_robot/rh_RFJ4_position_controller/command", "/my_robot/rh_LFJ1_position_controller/command",
                           "/my_robot/rh_LFJ2_position_controller/command", "/my_robot/rh_LFJ3_position_controller/command", "/my_robot/rh_LFJ4_position_controller/command", 
                           "/my_robot/rh_LFJ5_position_controller/command", "/my_robot/rh_THJ1_position_controller/command", "/my_robot/rh_THJ2_position_controller/command", 
                           "/my_robot/rh_THJ3_position_controller/command", "/my_robot/rh_THJ4_position_controller/command", "/my_robot/rh_THJ5_position_controller/command"]

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
