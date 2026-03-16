#!/usr/bin/env python3
import rospy
import csv
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState


class ShadowHandLoopback:
    def __init__(self):
        rospy.init_node('fake_shadowhand_bridge', anonymous=True)

        # 1. Joint Configuration 
        self.joint_names_ordered = [
            "rh_WRJ1", "rh_WRJ2", "rh_FFJ1", "rh_FFJ2", "rh_FFJ3", "rh_FFJ4", "rh_MFJ1",
            "rh_MFJ2", "rh_MFJ3", "rh_MFJ4", "rh_RFJ1", "rh_RFJ2", "rh_RFJ3", "rh_RFJ4", "rh_LFJ1",
            "rh_LFJ2", "rh_LFJ3", "rh_LFJ4", "rh_LFJ5", "rh_THJ1", "rh_THJ2", "rh_THJ3",
            "rh_THJ4", "rh_THJ5"
        ]
        
        # Internal state buffer (initialized to 0.0)
        # This acts as our "Perfect Robot" memory
        self.simulated_positions = {name: 0.0 for name in self.joint_names_ordered}

        # 2. Publishers & Subscribers
        self.state_topic = "/shadowhand_state_topic"
        self.julia_command_topic = "/shadowhand_command_topic"
        
        # Publish the "fake" state
        self.state_pub = rospy.Publisher(self.state_topic, JointState, queue_size=10)
        
        # Subscribe to Julia commands
        rospy.Subscriber(self.julia_command_topic, Float64MultiArray, self.command_callback)
        
        # Subscribe to real joint states ONLY to steal the timing/header
        rospy.Subscriber("/joint_states", JointState, self.sync_state_callback)

        rospy.loginfo("Loopback Node Started. Mimicking perfect hardware response.")

    def command_callback(self, msg):
        """
        Receives multi-array from Julia and updates the internal 
        'perfect' robot state.
        """
        # Note: We assume the incoming data maps to the 24 joints 
        # defined in joint_names_ordered
        if len(msg.data) < len(self.joint_names_ordered):
            return

        for i, name in enumerate(self.joint_names_ordered):
            self.simulated_positions[name] = msg.data[i]

    def sync_state_callback(self, msg):
        """
        Triggers every time the real robot (or gazebo) updates.
        We take the timestamp from 'msg' but the data from our command buffer.
        """
        sim_msg = JointState()
        sim_msg.header = msg.header # Mirror the exact timing/sequence of the real robot
        
        for name in self.joint_names_ordered:
            sim_msg.name.append(name)
            sim_msg.position.append(self.simulated_positions[name])
            sim_msg.velocity.append(0.0) # Instantaneous movement = 0 velocity in steady state

        self.state_pub.publish(sim_msg)


if __name__ == "__main__":
    try:
        node = ShadowHandLoopback()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass