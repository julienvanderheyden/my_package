#!/usr/bin/env python3

import rospy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from std_msgs.msg import Float64MultiArray, Float64

class CurrentListener:
    def __init__(self):
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
                        currents.append(value.value)  # Add the current value to the list

        # If we have found any currents, publish them
        if currents:
            currents_msg.data = currents
            self.current_pub.publish(currents_msg)

if __name__ == "__main__":
    try:
        node = CurrentListener()
        rospy.spin()  # Keep the node running
    except rospy.ROSInterruptException:
        pass
