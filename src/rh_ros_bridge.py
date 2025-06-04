#!/usr/bin/env python3
import rospy
import csv
import os
import argparse
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class ShadowHandBridge :
    def __init__(self, store_data, csv_filename, row_limit, rfj0_reset_value):
        

        rospy.init_node('shadowhand_ros_bridge', anonymous=True)

        ### STATE LISTENER : Listens to ShadowHand joint values, re-order them and post it on the right state topic ###

        # Joint order 
        self.joint_names_ordered = ["rh_WRJ1", "rh_WRJ2", "rh_FFJ1", "rh_FFJ2", "rh_FFJ3", "rh_FFJ4", "rh_MFJ1",
                                    "rh_MFJ2", "rh_MFJ3", "rh_MFJ4", "rh_RFJ1", "rh_RFJ2", "rh_RFJ3", "rh_RFJ4", "rh_LFJ1",
                                    "rh_LFJ2", "rh_LFJ3", "rh_LFJ4", "rh_LFJ5", "rh_THJ1", "rh_THJ2", "rh_THJ3",
                                    "rh_THJ4", "rh_THJ5"]
    
        # Publisher and subscriber
        self.state_topic = "/shadowhand_state_topic"
        self.state_pub = rospy.Publisher(self.state_topic, JointState, queue_size=10)

        rospy.Subscriber("/joint_states", JointState, self.state_callback)

        rospy.loginfo(f"Listening to /joint_states and publishing reordered positions on {self.state_topic}")

        ### TOPIC SPLITTER : Split the command sent by Julia to the correct joint command topics ### 

        # Publisher and subscriber
        self.julia_command_topic = "/shadowhand_command_topic"
        self.joints_command_topics = ["/sh_rh_wrj1_position_controller/command", "/sh_rh_wrj2_position_controller/command",
                           "/sh_rh_ffj0_position_controller/command", "/sh_rh_ffj3_position_controller/command", "/sh_rh_ffj4_position_controller/command", 
                           "/sh_rh_mfj0_position_controller/command", "/sh_rh_mfj3_position_controller/command", "/sh_rh_mfj4_position_controller/command", 
                           "/sh_rh_rfj0_position_controller/command", "/sh_rh_rfj3_position_controller/command", "/sh_rh_rfj4_position_controller/command", 
                           "/sh_rh_lfj0_position_controller/command", "/sh_rh_lfj3_position_controller/command", "/sh_rh_lfj4_position_controller/command",  
                           "/sh_rh_lfj5_position_controller/command", "/sh_rh_thj1_position_controller/command", "/sh_rh_thj2_position_controller/command", 
                           "/sh_rh_thj3_position_controller/command", "/sh_rh_thj4_position_controller/command", "/sh_rh_thj5_position_controller/command"]
        
        self.joint_command_publishers = [rospy.Publisher(topic, Float64, queue_size=10) for topic in self.joints_command_topics]
        rospy.Subscriber(self.julia_command_topic, Float64MultiArray, self.command_callback)

        # Joint limits and coupling

        self.joints_limits = [
                               [-0.69, 0.48], [-0.52, 0.71],
                               [0,1.57], [0, 1.57], [-0.26, 1.57], [-0.34, 0.34],
                               [0, 1.57], [0, 1.57], [-0.26, 1.57], [-0.34, 0.34],
                               [0, 1.57], [0, 1.57], [-0.26, 1.57], [-0.34, 0.34],
                               [0, 1.57], [0, 1.57], [-0.26, 1.57], [-0.34, 0.34], [0, 0.78],
                               [-0.26, 1.57], [-0.69, 0.69], [-0.2, 0.2], [0, 1.22], [-1.04, 1.04]
                            ]
        
        self.coupled_fingers = {2,5,8, 11}
        self.rfj0_reset_value = rfj0_reset_value

        # Logging rate
        self.rate = rospy.Rate(2)  # Print state 2 times per second
        self.joints_names = ["WRJ1", "WRJ2", "FFJ1", "FFJ2", "FFJ3", "FFJ4", "MFJ1", "MFJ2", "MFJ3", "MFJ4", "RFJ1", "RFJ2", "RFJ3", "RFJ4",
                            "LFJ1", "LFJ2", "LFJ3", "LFJ4", "LFJ5", "THJ1", "THJ2", "THJ3", "THJ4", "THJ5"]
        self.joint_states = [0.0] * len(self.joints_names)

        rospy.loginfo(f"Subscribed to: {self.julia_command_topic}")
        rospy.loginfo(f"Publishing to: {self.joints_command_topics}")

        ### CSV SAVING : store the sent and received joints values into a csv for further data analysis
        self.control_started = False
        self.store_data = store_data
        self.csv_file = csv_filename 
        self.row_limit = row_limit 
        self.row_count = 0
        
        if self.store_data:
            self.init_csv()
            rospy.loginfo(f"Saving joint values into {self.csv_file}")

    def init_csv(self):
        file_exists = os.path.isfile(self.csv_file)
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Timestamp"] + [f"cmd_{name}" for name in self.joints_names] + [f"state_{name}" for name in self.joints_names])

    def state_callback(self, msg):
        """
        Callback function to process JointState messages.
        It reorders the joint positions according to the joints_names_ordered.
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

        self.state_pub.publish(new_joint_state)
        #rospy.loginfo(f"state published on {self.output_topic}")

        ### SAVING VALUE INTO CSV 
        if self.store_data and self.control_started and self.row_count < self.row_limit: 
            self.write_csv(rospy.get_time(), self.joint_states, new_joint_state.position)
            self.row_count = self.row_count + 1

    def write_csv(self, timestamp, command_value, state_value):
        with open(self.csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp] + command_value + state_value)

    def command_callback(self, msg):
        """
        Callback function to process Float64MultiArray messages.
        It sends each command to the right topic. It also limits the joint values and handles the coupling between the joints. 
        """
        data = msg.data 

        # Ensure the received data length matches the number of publishers
        if len(data) <= len(self.joint_command_publishers):
            rospy.logwarn(f"Received {len(data)} values, but expected at least {len(self.joint_command_publishers)}. Ignoring message.")
            return

        # Publish each float to the corresponding topic
        j = 0
        for i in range(len(self.joint_command_publishers)):

            if self.joints_command_topics[i] == "/sh_rh_rfj0_position_controller/command" :
                self.joint_command_publishers[i].publish(Float64(self.rfj0_reset_value))
                j = j + 2

            elif i in self.coupled_fingers : # Check if joints are coupled
                j1_value = self.clamp(data[j], self.joints_limits[j][0], self.joints_limits[j][1])
                self.joint_states[j] = j1_value
                j = j +1
                j2_value = self.clamp(data[j], self.joints_limits[j][0], self.joints_limits[j][1])
                self.joint_states[j] = j2_value
                j = j+1
                limited_value = j1_value + j2_value # sum the two joints values
                self.joint_command_publishers[i].publish(Float64(limited_value))            

            else :
                limited_value = self.clamp(data[j], self.joints_limits[j][0], self.joints_limits[j][1])
                self.joint_states[j] = limited_value
                j = j+1
                self.joint_command_publishers[i].publish(Float64(limited_value))
        
        if not self.control_started :
            self.control_started = True

    def clamp(self, value, min_val, max_val):
        return max(min(value, max_val), min_val)
    
    def print_joint_states(self):
        while not rospy.is_shutdown():
            rospy.loginfo("\nJoint States:")
            for name, value in zip(self.joints_names, self.joint_states):
                rospy.loginfo(f"{name}: {value:.4f}")
            self.rate.sleep()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("store_data", type=str2bool, help="Boolean flag to enable or disable data storage")
    parser.add_argument("--csv_filename", type=str, default="shadowhand_joint_data.csv", help="Optional CSV filename")
    parser.add_argument("--row_limit", type=int, default=1000, help="Optional limit for the number of rows")
    parser.add_argument("--rfj0_reset_value", type=float, default=0.0, help="Value to reset RFJ0 joint to (default: 0.0)")

    args = parser.parse_args()

    try:
        node = ShadowHandBridge(args.store_data, args.csv_filename, args.row_limit, args.rfj0_reset_value)
        node.print_joint_states()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
