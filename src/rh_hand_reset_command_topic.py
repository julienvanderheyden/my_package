#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray

rospy.init_node("array_publisher")
pub = rospy.Publisher("/shadowhand_command_topic", Float64MultiArray, queue_size=10)

rate = rospy.Rate(10)  # 10 Hz
msg = Float64MultiArray()
msg.data = [0.0] * 24  # Array of 24 zeros

while not rospy.is_shutdown():
    pub.publish(msg)
    rate.sleep()
