#! /usr/bin/env python3
# coding: utf-8
import argparse
import errno
import ipaddress
import socket
import struct
import sys, os
import time
import numpy as np

####################################################################################################
# ROS stuff

import rospkg
import rospy
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from sensor_msgs.msg import JointState
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class ROSManager:
    subscriber_topic = None
    publisher_topic = None
    joint_command_size = None
    joint_state_size = None
    publisher = None
    joint_command_message = None
    rate = None
    new_msg = None

    def __init__(self, subscriber_topic, publisher_topic, joint_command_size, joint_state_size, rate):
        rospy.init_node('vmc_control')
        self.subscriber_topic = subscriber_topic
        self.publisher_topic = publisher_topic
        self.joint_command_size = joint_command_size
        self.joint_state_size = joint_state_size
        #rospy.Subscriber(subscriber_topic, JointState, self._subscriber_callback)
        rospy.Subscriber(subscriber_topic, Float64MultiArray, self._subscriber_callback)
        self.publisher = rospy.Publisher(publisher_topic, Float64MultiArray, queue_size=1)
        self.joint_command_message = Float64MultiArray()
        dim = MultiArrayDimension()
        dim.size = joint_command_size
        dim.stride = 1
        dim.label = "joint_effort"
        self.joint_command_message.layout.dim.append(dim)
        self.joint_command_message.data = [0.0] * joint_command_size
        self.rate = rospy.Rate(rate)

    def _subscriber_callback(self, msg):
        assert self.joint_state_size%2 == 0 
        # assert len(msg.position) == self.joint_state_size//2
        # assert len(msg.velocity) == self.joint_state_size//2
        self.new_msg = msg
        rospy.loginfo(msg)

        
class JointCommand:
    def __init__(self, sequence_number, timestamp, torques):
        self.sequence_number = sequence_number
        self.timestamp = timestamp
        self.torques = torques

STATE_WAITING = 0
STATE_WARMUP = 1
STATE_ACTIVE = 2
STATE_STOPPED = 3




####################################################################################################



# def send_recv_send_recv_wait(socket_manager, ros_manager, set_zero=False):
#     if ros_manager.new_msg is not None: # New msg received from ROS subscriber
#         forward_state_to_julia(socket_manager, ros_manager)
#     command = socket_manager.recv_joint_command() 
#     if command is not None: # New torque command received from julia
#         if set_zero:
#             ros_manager.joint_command_message.data = 0 * command.torques
#         else:
#             ros_manager.joint_command_message.data = command.torques
#         ros_manager.publisher.publish(ros_manager.joint_command_message) # Publish via ROS
#     ros_manager.rate.sleep()


def loop_active(trajectory, ros_manager):
    print("State: ACTIVE")
    i = 0
    trajectory_done = False
    pos_limit_index = int((len(trajectory[0]) -1 )/2)+1
    while not rospy.is_shutdown():
        if ros_manager.new_msg is not None: # New msg received from ROS subscriber
            print(ros_manager.new_msg)

        if trajectory_done:
            ros_manager.joint_command_message.data = np.zeros(pos_limit_index-1)
        else :
            ros_manager.joint_command_message.data = trajectory[i][1:pos_limit_index]

        ros_manager.publisher.publish(ros_manager.joint_command_message) # Publish via ROS
        i = i + 1
        if i >= trajectory.shape[0] :
            i = 0
            trajectory_done = not trajectory_done

        ros_manager.rate.sleep()
        # command = socket_manager.command_stream.readline()
        # if command == "":
        #     send_recv_send_recv_wait(socket_manager, ros_manager)
        # elif command == "STOP\n":
        #     return STATE_STOPPED
        # else:
        #     print(f"Unexpected command in state ACTIVE: \"{command}\".")
        #     return STATE_STOPPED

####################################################################################################
# Main

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("joint_command_size", type=int)
    parser.add_argument("joint_commands_topic", type=str)
    parser.add_argument("joint_state_topic", type=str)
    parser.add_argument("trajectory_file", type=str)
    #parser.add_argument("--rate", type=float, default=1000)
    parser.add_argument("--joint_state_size", default=None)
    parser.add_argument("--listen_port", type=int, default=25342)
    parser.add_argument("--listen_ip", type=str, default="127.0.0.1")
    parser.add_argument("--auto-restart", type=bool, default=True)

    args = parser.parse_args()
    joint_command_size = args.joint_command_size
    joint_state_size = args.joint_state_size if args.joint_state_size is not None else 2 * joint_command_size 
    joint_commands_topic = args.joint_commands_topic
    joint_state_topic = args.joint_state_topic
    listen_port = args.listen_port
    listen_ip = ipaddress.ip_address(args.listen_ip)

    rospack = rospkg.RosPack()
    package_path = rospack.get_path('my_package')  
    csv_path = package_path + "/config/" +args.trajectory_file
    trajectory = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    rate = 1/(trajectory[1][0] - trajectory[0][0])

    # Setup ROS
    # rospy.init_node('vmc_control')
    ros_manager = ROSManager(
        joint_state_topic,
        joint_commands_topic,
        joint_command_size,
        joint_state_size,
        rate
    )


    while not rospy.is_shutdown():
        state = STATE_ACTIVE
        while not rospy.is_shutdown():
            try:
                if state == STATE_ACTIVE:
                    state = loop_active(trajectory, ros_manager)
                elif state == STATE_STOPPED:
                    break # Loop back to waiting for a new connection
                else:
                    raise Exception(f"Invalid state: {state}")
            except Exception as e:
                print(f"Unhandled Exception: {e}")
            if not args.auto_restart:
                break


