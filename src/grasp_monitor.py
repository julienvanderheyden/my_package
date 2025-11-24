#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int32
from visualization_msgs.msg import MarkerArray
import math
import sys

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
        
        # Display control
        self.display_initialized = False
        self.last_display_time = rospy.Time(0)
        self.display_rate = 10.0  # Hz - update display 10 times per second
        
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
        rospy.loginfo("Waiting for grasp to begin...\n")
    
    def grasp_callback(self, msg):
        """
        Callback for grasp timing events
        """
        if msg.data == 1 and not self.grasping:
            # Grasp started
            self.grasping = True
            self.grasp_failed = False
            self.display_initialized = False
            
            # Clear screen and print header
            sys.stdout.write("\033[2J\033[H")  # Clear screen and move cursor to top
            sys.stdout.flush()
            
            print("=" * 80)
            print("                         GRASP STARTED")
            print("=" * 80)
            print("Monitoring tactile feedback...\n")
            
        elif msg.data == 0 and self.grasping:
            # Grasp ended
            self.grasping = False
            
            # Clear the dynamic display area
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
            
            print("\n" + "=" * 80)
            print("                          GRASP ENDED")
            print("=" * 80)
            
            # Determine grasp outcome
            if self.grasp_failed:
                print(">>> GRASP RESULT: FAILURE - Contact was lost during grasp <<<")
            else:
                print(">>> GRASP RESULT: SUCCESS - Contact maintained throughout grasp <<<")
            
            print("=" * 80 + "\n")
            print("Waiting for next grasp...\n")
            
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
        
        for i, marker in enumerate(msg.markers):
            # Check if taxel has contact (alpha = 1.0)
            if marker.color.a >= 0.99:
                # Get the second point (arrow head position)
                if len(marker.points) >= 2:
                    arrow_head = marker.points[1]
                    magnitude = self.compute_vector_magnitude(arrow_head)
                    total_magnitude += magnitude

        # Store the aggregated magnitude for this finger
        self.finger_magnitudes[finger_name] = total_magnitude
        
        # Check if all fingers have been updated
        if all(data is not None for data in self.tactile_data.values()):
            self.check_contact_state_and_display()
    
    def check_contact_state_and_display(self):
        """
        Check contact state and display verbose information
        """
        if not self.grasping or self.grasp_failed:
            return
        
        # Rate limit the display updates
        current_time = rospy.Time.now()
        if (current_time - self.last_display_time).to_sec() < (1.0 / self.display_rate):
            return
        
        self.last_display_time = current_time
        
        # Check if any finger has contact
        has_contact = any(mag > 0.0 for mag in self.finger_magnitudes.values())
        
        # Display verbose output
        self.display_verbose_output()
        
        # Update contact state
        if not has_contact:
            print("\n" + "!" * 80)
            print("                    CONTACT LOST - GRASP FAILED")
            print("!" * 80 + "\n")
            self.grasp_failed = True
    
    def display_verbose_output(self):
        """
        Display beautiful verbose output for analysis with static positioning
        """
        if not self.display_initialized:
            # First time: just print the display
            self.display_initialized = True
            self.print_display()
        else:
            # Move cursor up to overwrite previous display
            # We need to move up 10 lines (header + 5 fingers + footer + separators)
            sys.stdout.write("\033[10A")  # Move cursor up 10 lines
            sys.stdout.flush()
            self.print_display()
    
    def print_display(self):
        """
        Print the actual display content
        """
        # Compute total force
        total_force = sum(self.finger_magnitudes.values())
        
        # Finger name mapping for display
        finger_names = {
            'ff': 'Forefinger',
            'mf': 'Middle    ',
            'rf': 'Ring      ',
            'lf': 'Little    ',
            'th': 'Thumb     '
        }
        
        # Build the display
        lines = []
        lines.append("-" * 80)
        lines.append("  TACTILE FEEDBACK - Real-time Monitor")
        lines.append("-" * 80)
        
        # Display individual finger magnitudes
        for finger_key in ['ff', 'mf', 'rf', 'lf', 'th']:
            magnitude = self.finger_magnitudes[finger_key]
            bar_length = int(magnitude * 1000)  # Scale for visualization
            bar = "#" * min(bar_length, 40)  # Cap at 40 chars
            
            line = "  %s: %8.4f  %s" % (finger_names[finger_key], magnitude, bar)
            lines.append(line.ljust(80))  # Pad to 80 chars to clear previous content
        
        lines.append("  " + "-" * 76)
        lines.append(("  TOTAL FORCE: %.4f" % total_force).ljust(80))
        lines.append("-" * 80)
        
        # Print all lines
        for line in lines:
            print(line)
        
        sys.stdout.flush()
    
    def reset_state(self):
        """
        Reset tactile data after grasp ends
        """
        for key in self.tactile_data:
            self.tactile_data[key] = None
        for key in self.finger_magnitudes:
            self.finger_magnitudes[key] = 0.0
        self.display_initialized = False
    
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