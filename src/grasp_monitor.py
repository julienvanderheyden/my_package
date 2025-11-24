#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32
from visualization_msgs.msg import MarkerArray

class GraspSuccessMonitor:
    def __init__(self):
        # Initialize the node
        rospy.init_node('grasp_success_monitor', anonymous=True)
        
        # Grasp state
        self.grasping = False
        self.grasp_failed = False
        
        # Tactile data storage for each finger
        self.tactile_data = {
            'ff': None,  # Forefinger
            'mf': None,  # Middle finger
            'rf': None,  # Ring finger
            'lf': None,  # Little finger
            'th': None   # Thumb
        }
        
        # Subscribe to grasp timing
        rospy.Subscriber('/grasp_timing', Int32, self.grasp_callback)
        
        # Subscribe to tactile topics
        self.tactile_subscribers = {
            'ff': rospy.Subscriber('/rh/publish_hand_mst_markers/rh_ff_arrows', 
                                   MarkerArray, 
                                   lambda msg: self.tactile_callback(msg, 'ff')),
            'mf': rospy.Subscriber('/rh/publish_hand_mst_markers/rh_mf_arrows', 
                                   MarkerArray, 
                                   lambda msg: self.tactile_callback(msg, 'mf')),
            'rf': rospy.Subscriber('/rh/publish_hand_mst_markers/rh_rf_arrows', 
                                   MarkerArray, 
                                   lambda msg: self.tactile_callback(msg, 'rf')),
            'lf': rospy.Subscriber('/rh/publish_hand_mst_markers/rh_lf_arrows', 
                                   MarkerArray, 
                                   lambda msg: self.tactile_callback(msg, 'lf')),
            'th': rospy.Subscriber('/rh/publish_hand_mst_markers/rh_th_arrows', 
                                   MarkerArray, 
                                   lambda msg: self.tactile_callback(msg, 'th'))
        }
        
        rospy.loginfo("Grasp Success Monitor node started")
    
    def grasp_callback(self, msg):
        """
        Callback for grasp timing events
        """
        if msg.data == 1 and not self.grasping:
            # Grasp started
            self.grasping = True
            self.grasp_failed = False
            rospy.loginfo("=== GRASP STARTED ===")
            rospy.loginfo("Monitoring tactile feedback...")
            
        elif msg.data == 0 and self.grasping:
            # Grasp ended
            self.grasping = False
            rospy.loginfo("=== GRASP ENDED ===")
            
            # Determine grasp outcome
            if self.grasp_failed:
                rospy.logwarn("GRASP RESULT: FAILURE - Contact was lost during grasp")
            else:
                rospy.loginfo("GRASP RESULT: SUCCESS - Contact maintained throughout grasp")
            
            # Reset state
            self.reset_state()
    
    def tactile_callback(self, msg, finger_name):
        """
        Callback for tactile sensor data
        """
        # Only process tactile data during active grasp
        if not self.grasping:
            return
        
        # Store tactile data
        self.tactile_data[finger_name] = msg
        
        # Check if all fingers have data
        if all(data is not None for data in self.tactile_data.values()):
            self.check_contact_state()
    
    def check_contact_state(self):
        """
        Check if any taxel across all fingers has contact
        """
        if not self.grasping or self.grasp_failed:
            return
        
        has_contact = False
        
        # Check all fingers for contact
        for finger_name, marker_array in self.tactile_data.items():
            if marker_array is None:
                continue
            
            # Check each marker (taxel) in the array
            for marker in marker_array.markers:
                # Contact is detected when alpha value is 1.0
                if marker.color.a >= 0.99:  # Using >= 0.99 for floating point comparison
                    has_contact = True
                    break
            
            if has_contact:
                break
        

        if not has_contact:
            # Contact was previously detected but now lost
            rospy.logwarn("Contact lost! Grasp marked as FAILURE")
            self.grasp_failed = True

    def reset_state(self):
        """
        Reset tactile data after grasp ends
        """
        for key in self.tactile_data:
            self.tactile_data[key] = None
    
    def run(self):
        """
        Keep the node running
        """
        rospy.spin()

if __name__ == '__main__':
    try:
        node = GraspSuccessMonitor()
        node.run()
    except rospy.ROSInterruptException:
        pass