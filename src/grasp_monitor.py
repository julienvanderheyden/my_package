#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int32
from visualization_msgs.msg import MarkerArray
import math

class GraspSuccessMonitor:
    def __init__(self):
        # Initialize the node
        rospy.init_node('grasp_success_monitor', anonymous=True)
        
        # Grasp state
        self.grasping = False
        self.grasp_failed = False
        
        # Force magnitude storage for each finger
        self.finger_magnitudes = {
            'ff': 0.0,  # Forefinger
            'mf': 0.0,  # Middle finger
            'rf': 0.0,  # Ring finger
            'lf': 0.0,  # Little finger
            'th': 0.0   # Thumb
        }
        
        # Tactile data storage for each finger
        self.tactile_data = {
            'ff': None,
            'mf': None,
            'rf': None,
            'lf': None,
            'th': None
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
            self.contact_detected = False
            rospy.loginfo("\n" + "="*80)
            rospy.loginfo("                         GRASP STARTED")
            rospy.loginfo("="*80)
            rospy.loginfo("Monitoring tactile feedback...\n")
            
        elif msg.data == 0 and self.grasping:
            # Grasp ended
            self.grasping = False
            rospy.loginfo("\n" + "="*80)
            rospy.loginfo("                          GRASP ENDED")
            rospy.loginfo("="*80)
            
            # Determine grasp outcome
            if self.grasp_failed:
                rospy.logwarn(">>> GRASP RESULT: FAILURE - Contact was lost during grasp <<<")
            else:
                rospy.loginfo(">>> GRASP RESULT: SUCCESS - Contact maintained throughout grasp <<<")
            
            rospy.loginfo("="*80 + "\n")
            
            # Reset state
            self.reset_state()
    
    def compute_vector_magnitude(self, point):
        """
        Compute the magnitude (norm) of a 3D vector
        """
        return math.sqrt(point.x**2 + point.y**2 + point.z**2)
    
    def tactile_callback(self, msg, finger_name):
        """
        Callback for tactile sensor data
        """
        # Only process tactile data during active grasp
        if not self.grasping:
            return
        
        # Store tactile data
        self.tactile_data[finger_name] = msg
        
        # Compute force magnitude for this finger
        total_magnitude = 0.0
        active_taxels = 0
        taxel_magnitudes = []
        
        for i, marker in enumerate(msg.markers):
            # Check if taxel has contact (alpha = 1.0)
            if marker.color.a >= 0.99:
                # Get the second point (arrow head position)
                if len(marker.points) >= 2:
                    arrow_head = marker.points[1]
                    magnitude = self.compute_vector_magnitude(arrow_head)
                    total_magnitude += magnitude
                    active_taxels += 1
                    taxel_magnitudes.append((i, magnitude))
        
        # Store the aggregated magnitude for this finger
        self.finger_magnitudes[finger_name] = total_magnitude
        
        # Check if all fingers have been updated
        if all(data is not None for data in self.tactile_data.values()):
            self.check_contact_state_and_display(finger_name, active_taxels, taxel_magnitudes)
    
    def check_contact_state_and_display(self, updated_finger, active_taxels, taxel_magnitudes):
        """
        Check contact state and display verbose information
        """
        if not self.grasping or self.grasp_failed:
            return
        
        # Check if any finger has contact
        has_contact = any(mag > 0.0 for mag in self.finger_magnitudes.values())
        
        # Display verbose output
        self.display_verbose_output()
        
        # Update contact state
        if not has_contact:
            rospy.logwarn("CONTACT LOST - GRASP FAILED")
            self.grasp_failed = True
    
    def display_verbose_output(self):
        """
        Display beautiful verbose output for analysis
        """
        
        # Display individual finger magnitudes
        rospy.loginfo("-" * 80 + "\n")
        rospy.loginfo("  Finger Force Magnitudes:")
        for finger_key in ['ff', 'mf', 'rf', 'lf', 'th']:
            magnitude = self.finger_magnitudes[finger_key]
            bar_length = int(magnitude * 1000)  # Scale for visualization
            bar = "#" * min(bar_length, 50)  # Cap at 50 chars
            
            rospy.loginfo("    %s: %8.4f  %s" % (
                finger_key, 
                magnitude, 
                bar
            ))
        rospy.loginfo("-" * 80 + "\n")
    
    def reset_state(self):
        """
        Reset tactile data after grasp ends
        """
        for key in self.tactile_data:
            self.tactile_data[key] = None
        for key in self.finger_magnitudes:
            self.finger_magnitudes[key] = 0.0
    
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