#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState
import time

def fake_joint_state_publisher():
    rospy.init_node('fake_joint_state_publisher', anonymous=True)
    pub = rospy.Publisher('/joint_states', JointState, queue_size=10)

    # Define joint names in the same order expected by ShadowHandBridge
    joint_names = ["rh_WRJ1", "rh_WRJ2", "rh_FFJ1", "rh_FFJ2", "rh_FFJ3", "rh_FFJ4", "rh_MFJ1",
                   "rh_MFJ2", "rh_MFJ3", "rh_MFJ4", "rh_RFJ1", "rh_RFJ2", "rh_RFJ3", "rh_RFJ4", "rh_LFJ1",
                   "rh_LFJ2", "rh_LFJ3", "rh_LFJ4", "rh_LFJ5", "rh_THJ1", "rh_THJ2", "rh_THJ3",
                   "rh_THJ4", "rh_THJ5"]

    rate = rospy.Rate(125)  # 125 Hz : shadow hand frequency
    while not rospy.is_shutdown():
        js = JointState()
        js.header.stamp = rospy.Time.now()
        js.name = joint_names
        js.position = [0.0] * len(joint_names)
        js.velocity = [0.0] * len(joint_names)
        js.effort = [0.0] * len(joint_names)

        pub.publish(js)
        rate.sleep()

if __name__ == '__main__':
    try:
        fake_joint_state_publisher()
    except rospy.ROSInterruptException:
        pass
