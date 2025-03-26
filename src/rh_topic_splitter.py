#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray

class DynamicRouter:
    def __init__(self):
        rospy.init_node('dynamic_router', anonymous=True)

        self.sub_topic = "/shadowhand_command_topic"
        self.pub_topics = ["/sh_rh_wrj1_position_controller/command", "/sh_rh_wrj2_position_controller/command",
                           "/sh_rh_ffj0_position_controller/command", "/sh_rh_ffj3_position_controller/command", "/sh_rh_ffj4_position_controller/command", 
                           "/sh_rh_mfj0_position_controller/command", "/sh_rh_mfj3_position_controller/command", "/sh_rh_mfj4_position_controller/command", 
                           "/sh_rh_rfj0_position_controller/command", "/sh_rh_rfj3_position_controller/command", "/sh_rh_rfj4_position_controller/command", 
                           "/sh_rh_lfj0_position_controller/command", "/sh_rh_lfj3_position_controller/command", "/sh_rh_lfj4_position_controller/command",  
                           "/sh_rh_lfj5_position_controller/command", "/sh_rh_thj1_position_controller/command", "/sh_rh_thj2_position_controller/command", 
                           "/sh_rh_thj3_position_controller/command", "/sh_rh_thj4_position_controller/command", "/sh_rh_thj5_position_controller/command"]

        self.publishers = [rospy.Publisher(topic, Float64, queue_size=10) for topic in self.pub_topics]
        rospy.Subscriber(self.sub_topic, Float64MultiArray, self.callback)

        self.joints_limits = [
                               [-0.69, 0.48], [-0.52, 0.71],
                               [0,1.57], [0, 1.57], [-0.26, 1.57], [-0.34, 0.34],
                               [0, 1.57], [0, 1.57], [-0.26, 1.57], [-0.34, 0.34],
                               [0, 1.57], [0, 1.57], [-0.26, 1.57], [-0.34, 0.34],
                               [0, 1.57], [0, 1.57], [-0.26, 1.57], [-0.34, 0.34], [0, 0.78],
                               [-0.26, 1.57], [-0.69, 0.69], [-0.2, 0.2], [0, 1.22], [-1.04, 1.04]
                            ]
        
        self.joints_names = ["WRJ1", "WRJ2", "FFJ1", "FFJ2", "FFJ3", "FFJ4", "MFJ1", "MFJ2", "MFJ3", "MFJ4", "RFJ1", "RFJ2", "RFJ3", "RFJ4",
                            "LFJ1", "LFJ2", "LFJ3", "LFJ4", "LFJ5", "THJ1", "THJ2", "THJ3", "THJ4", "THJ5"]

        self.coupled_fingers = {2,5,8, 11}

        self.rate = rospy.Rate(2)  # Print state 2 times per second
        self.joint_states = [0.0] * len(self.joints_names)

        rospy.loginfo(f"Subscribed to: {self.sub_topic}")
        rospy.loginfo(f"Publishing to: {self.pub_topics}")

    def clamp(self, value, min_val, max_val):
        return max(min(value, max_val), min_val)

    def callback(self, msg):
        data = msg.data  # Extract array from Float64MultiArray

        # Ensure the received data length matches the number of publishers
        if len(data) <= len(self.publishers):
            rospy.logwarn(f"Received {len(data)} values, but expected at least {len(self.publishers)}. Ignoring message.")
            return

        # Publish each float to the corresponding topic
        j = 0
        for i in range(len(data)):
            if i in self.coupled_fingers : 
                j1_value = self.clamp(data[j], self.joints_limits[j][0], self.joints_limits[j][1])
                self.joint_states[j] = j1_value
                j = j +1
                j2_value = self.clamp(data[j], self.joints_limits[j][0], self.joints_limits[j][1])
                self.joint_states[j] = j2_value
                j = j+1
                limited_value = j1_value + j2_value # sum the two joints values
                self.publishers[i].publish(Float64(limited_value))            

            else :
                limited_value = self.clamp(data[j], self.joints_limits[j][0], self.joints_limits[j][1])
                self.joint_states[j] = limited_value
                j = j+1
                self.publishers[i].publish(Float64(limited_value))


    def print_joint_states(self):
        while not rospy.is_shutdown():
            rospy.loginfo("\nJoint States:")
            for name, value in zip(self.joints_names, self.joint_states):
                rospy.loginfo(f"{name}: {value:.4f}")
            self.rate.sleep()

if __name__ == "__main__":
    try:
        node = DynamicRouter()
        node.print_joint_states()
        rospy.spin()  # Keep the node running
    except rospy.ROSInterruptException:
        pass
