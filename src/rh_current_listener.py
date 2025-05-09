#!/usr/bin/env python3

import rospy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from std_msgs.msg import Float64MultiArray, Float64
import csv
import os
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class CurrentListener:
    def __init__(self, store_data, csv_filename):
        # Initialize the ROS node
        rospy.init_node("current_listener", anonymous=True)

        # Get the list of desired joint names from parameter server
        self.motor_names = ["rh SRDMotor WRJ1", "rh SRDMotor WRJ2", "rh SRDMotor FFJ0", "rh SRDMotor FFJ3", "rh SRDMotor FFJ4", "rh SRDMotor MFJ0",
                            "rh SRDMotor MFJ3", "rh SRDMotor MFJ4", "rh SRDMotor RFJ0", "rh SRDMotor RFJ3", "rh SRDMotor RFJ4", "rh SRDMotor LFJ0",
                            "rh SRDMotor LFJ3", "rh SRDMotor LFJ4", "rh SRDMotor LFJ5", "rh SRDMotor THJ1", "rh SRDMotor THJ2", "rh SRDMotor THJ3",
                            "rh SRDMotor THJ4", "rh SRDMotor THJ5"]
        

        # Get the output topic name
        self.output_topic = "/shadowhand_currents"

        # Publisher to send reordered joint positions
        self.current_pub = rospy.Publisher(self.output_topic, Float64MultiArray, queue_size=10)

        # Subscriber to listen to joint states
        rospy.Subscriber("/diagnostics", DiagnosticArray, self.callback)

        rospy.loginfo(f"Listening to /diagnostics and publishing currents on {self.output_topic}")

        self.store_data = store_data
        self.csv_file = csv_filename 

        if self.store_data:
            self.init_csv()
            rospy.loginfo(f"Saving joint values into {self.csv_file}")


    def callback(self, msg):
        """
        Callback function to process DiagnosticArray messages.
        It extracts the current and send them.
        """
        currents_msg = Float64MultiArray()
        currents = []
                
        # Loop through all the status items in the DiagnosticArray
        for status in msg.status:
            if status.name in self.motor_names:  # Check if the motor is in our list

                for value in status.values:
                    if value.key == "Measured Current":
                        currents.append(float(value.value))  # Add the current value to the list

        # If we have found any currents, publish them
        if currents:
            currents_msg.data = currents
            self.current_pub.publish(currents_msg)

            if self.store_data : 
                self.write_csv(rospy.get_time(), currents)

    def init_csv(self):
        file_exists = os.path.isfile(self.csv_file)
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Timestamp"] + [f"{name}" for name in self.motor_names])

    def write_csv(self, timestamp, currents):
        with open(self.csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp] + currents)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("store_data", type=str2bool, help="Boolean flag to enable or disable data storage")
    parser.add_argument("--csv_filename", type=str, default="shadowhand_current_data.csv", help="Optional CSV filename")
    args = parser.parse_args()

    try:
        node = CurrentListener(args.store_data, args.csv_filename)
        rospy.spin()  # Keep the node running
    except rospy.ROSInterruptException:
        pass
