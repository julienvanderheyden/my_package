#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState

class StateListener:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node("state_listener", anonymous=True)

        # Get the list of desired joint names from parameter server
        self.joint_names_ordered = ["rh_WRJ1", "rh_WRJ2", "rh_FFJ1", "rh_FFJ2", "rh_FFJ3", "rh_FFJ4", "rh_MFJ1",
                                    "rh_MFJ2", "rh_MFJ3", "rh_MFJ4", "rh_RFJ1", "rh_RFJ2", "rh_RFJ3", "rh_RFJ4", "rh_LFJ1",
                                    "rh_LFJ2", "rh_LFJ3", "rh_LFJ4", "rh_LFJ5", "rh_THJ1", "rh_THJ2", "rh_THJ3",
                                    "rh_THJ4", "rh_THJ5"]
        

        # Get the output topic name
        self.output_topic = "/shadowhand_state_topic"

        # Publisher to send reordered joint positions
        self.pub = rospy.Publisher(self.output_topic, JointState, queue_size=10)

        # Subscriber to listen to joint states
        rospy.Subscriber("/my_robot/joint_states", JointState, self.callback)

        rospy.loginfo(f"Listening to /joint_states and publishing reordered positions on {self.output_topic}")

    def callback(self, msg):
        """
        Callback function to process JointState messages.
        It reorders the joint positions according to the target_joint_order.
        """
        new_joint_state = JointState()
        new_joint_state.header = msg.header  # Keep the same timestamp

        # Create a mapping from joint name to its index in the original message
        joint_index_map = {name: i for i, name in enumerate(msg.name)}

        # Reorder joint data based on the desired order
        for joint_name in self.joint_names_ordered:
            if joint_name in joint_index_map:
                index = joint_index_map[joint_name]
                new_joint_state.name.append(joint_name)
                new_joint_state.position.append(msg.position[index])
                if len(msg.velocity) > 0:
                    new_joint_state.velocity.append(msg.velocity[index])
            else:
                rospy.logwarn(f"Joint '{joint_name}' not found in the received JointState message.")

        self.pub.publish(new_joint_state)
        rospy.loginfo(f"state published on {self.output_topic}")

if __name__ == "__main__":
    try:
        node = StateListener()
        rospy.spin()  # Keep the node running
    except rospy.ROSInterruptException:
        pass
