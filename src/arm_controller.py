#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
import geometry_msgs.msg
from math import pi
from tf.transformations import quaternion_from_euler
from std_msgs.msg import Int32
from copy import deepcopy
import numpy as np
import tf.transformations as tf


class UR10eMoveItController:
    def __init__(self, group_name="right_arm"):
        # Initialize MoveIt Commander and ROS node
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("ur10e_cartesian_goal", anonymous=True)


        # Initialize Move Group
        self.move_group = moveit_commander.MoveGroupCommander(group_name)

        # Publishers and subscribers
        self.sync_sub = rospy.Subscriber(
            "/ros_julia_synchronization", Int32, self.sync_callback
        )
        self.sync_pub = rospy.Publisher(
            "/ros_julia_synchronization", Int32, queue_size=10
        )

        self.timing_pub = rospy.Publisher(
            "/grasp_timing", Int32, queue_size=10
        )

        self.preshape_pub = rospy.Publisher("/preshape", Int32, queue_size=10)

        ### THIS SHOULD BE CHANGED FOR EACH TEST ###
    
        #grasp type 
        self.grasp_type = 1 # 1 : medium wrap, 2 : power sphere, 3 : lateral pinch

        self.parameters = [0.01, 0.015, 0.02, 0.0225, 0.025, 0.0275, 0.03, 0.035, 0.04] # cylinders
        # self.parameters =  [0.019, 0.019, 0.03, 0.035, 0.04] # spheres
        # self.parameters = [0.019, 0.019, 0.03, 0.035, 0.04] # cubes
        # self.parameters = [[0.025, 0.001], [0.025, 0.006],[0.025, 0.013], [0.0375, 0.001], 
        # [0.0375, 0.006], [0.0375, 0.013], [0.05, 0.001], [0.05, 0.006], [0.05, 0.013]]

        # Predefined positions (x, y, z) 
        if self.grasp_type == 2 : #power sphere
            self.reference_positions = [
                [0.85, 0.173, 1.07], #home position
                [1.218, 0.738, 0.866], #left most position
                [1.218, 0.499, 0.866],
                [1.218, 0.258, 0.866],
                [1.218, 0.023, 0.866], 
                [1.218, -0.219, 0.866]] # right most position

        else : #medium wrap, lateral pinch
            self.reference_positions = [
                [0.85, 0.173, 1.07], # 1 : home position
                [1.218, 0.738, 0.866], # 2 : leftmost position
                [1.218, 0.618, 0.866], # 3
                [1.218, 0.499, 0.866], # 4 
                [1.218, 0.379, 0.866], # 5 
                [1.218, 0.258, 0.866], # 6 
                [1.218, 0.139, 0.866], # 7
                [1.218, 0.023, 0.866], # 8
                [1.218, -0.098, 0.866], # 9 
                [1.218, -0.219, 0.866], # 10 : rightmost position
            ]

        self.position_sigma = 0.0
        self.orientation_sigma = 0.0

        ##############################################

        # Compute palm position based on the reference points
        self.positions = [self.reference_positions[0]]  # Start with the home position

        for i in range(1, len(self.parameters) + 1) :
            palm_position = self.compute_palm_position(self.reference_positions[i], self.parameters[i-1], i)
            self.positions.append(palm_position)

        if self.grasp_type != 2: #medium wrap, lateral pinch
            self.orientations = np.repeat([[pi, -pi/2, 0]], len(self.positions), axis=0).tolist()

        else : #power sphere
            self.orientations = np.repeat([[pi/2, 0, pi/2]], len(self.positions), axis=0).tolist()

        self.planning_frame = self.move_group.get_planning_frame()
        self.eef_link = self.move_group.get_end_effector_link()
        self.current_pose = 0 

        rospy.loginfo(f"Planning frame: {self.planning_frame}")
        rospy.loginfo(f"End effector link: {self.eef_link}")

        self.reach_cartesian(self.positions[0], self.orientations[0]) #going to home position
        rospy.loginfo("UR10eMoveItController initialized and listening for commands...")

    def compute_palm_position(self, ref_position, parameter, i):
        if self.grasp_type == 1:  # medium wrap
            radius = parameter  # radius of the cylinder
            alpha = 1.31       # angle between thumb and palm at preshape

            # z position: small vertical offset from cylinder center
            z = ref_position[2] + 0.09  

            if radius >= 0.015 : 
                # y position: palm tangent to the cylinder
                y = ref_position[1] - radius - 0.03 
                # x position: thumb tangent to the cylinder
                # x = ref_position[0] - (radius + 0.01) * (np.cos(alpha/2) / np.sin(alpha/2)) - 0.04
                x = ref_position[0] - 0.095 # when doing tests with fixed x offset

            else : 
                # y position : 2cm offset 
                y = ref_position[1] - 0.03
                # x position : 9.5 cm offset 
                x = ref_position[0] - 0.095

            return [x, y, z]

        elif self.grasp_type == 2:  # power sphere
            radius = parameter  # radius of the sphere
            stand_height = 0.17  # height of the stand supporting the sphere
            depth_ratio = 4/3 # how high the ball goes above the stand

            
            if radius >= 0.02 :
                # z position: above the sphere surface
                z = ref_position[2] + stand_height + depth_ratio*radius #+ 0.021 
            
            else :
                # z position : fixed offset above the sphere center
                z = ref_position[2] + stand_height + (1 - depth_ratio)*radius + 0.11

            # y position: aligned with sphere center
            y = ref_position[1] - 0.03

            # x position: aligned to get the center of the palm centered with the sphere center
            x = ref_position[0] - 0.083 

            return [x, y, z]
        
        else : # lateral pinch 
            # parameters do not affect the position of the hand
            if i <= 5 :
                support_height = 0.2  # height of the stand supporting the object
            else : 
                support_height = 0.23

            # z position : 
            palm_knuckle_dist = 0.033
            finger_width = 0.018
            z = ref_position[2] + support_height  - palm_knuckle_dist - finger_width/2 #- 0.05

            # y position :
            y = ref_position[1] - 0.03

            # x position :
            x = ref_position[0] - 0.125

            return [x, y, z]

        return None

    def get_current_pose(self):
        """Return the current end-effector pose (geometry_msgs/Pose)."""
        return self.move_group.get_current_pose(self.eef_link).pose

    def reach_cartesian(self, position, orientation, eef_step=0.005):
        """
        Move the end-effector (palm frame) to a target position using a smooth Cartesian path.
        Internally, this computes the corresponding flange pose from the known fixed transform.
        Args:
            position: (x, y, z) desired palm position in meters (in base frame)
            orientation: (roll, pitch, yaw) desired palm orientation in radians (in base frame)
            eef_step: distance between waypoints in meters
        """

        # --- Fixed transform (ra_flange -> rh_palm) measured with tf_echo ---
        # Translation in meters
        t_flange_palm = [0.247, 0.000, 0.010]
        # Rotation (in radians) from tf_echo: [-1.575, 0.000, -1.563]
        r_flange_palm = [-1.575, 0.000, -1.563]
        T_flange_palm = tf.euler_matrix(*r_flange_palm)
        T_flange_palm[0:3, 3] = t_flange_palm

        # Compute the inverse transform: palm -> flange
        T_palm_flange = tf.inverse_matrix(T_flange_palm)

        # --- Desired palm pose (input) ---
        T_base_palm = tf.euler_matrix(*orientation)
        T_base_palm[0:3, 3] = position

        # --- Compute corresponding flange pose in base frame ---
        T_base_flange = T_base_palm @ T_palm_flange

        flange_position = tf.translation_from_matrix(T_base_flange)
        flange_quat = tf.quaternion_from_matrix(T_base_flange)
        flange_rpy = tf.euler_from_quaternion(flange_quat)

        # --- Now perform the actual Cartesian motion using the flange pose ---
        current_pose = deepcopy(self.get_current_pose())
        target_pose = deepcopy(current_pose)

        target_pose.position.x = flange_position[0]
        target_pose.position.y = flange_position[1]
        target_pose.position.z = flange_position[2]

        q = tf.quaternion_from_euler(*flange_rpy)
        target_pose.orientation.x = q[0]
        target_pose.orientation.y = q[1]
        target_pose.orientation.z = q[2]
        target_pose.orientation.w = q[3]

        waypoints = [target_pose]

        (plan, fraction) = self.move_group.compute_cartesian_path(
            waypoints,
            eef_step=eef_step,
            jump_threshold=0.0
        )

        if fraction < 1.0:
            rospy.logwarn(f"Cartesian path only {fraction*100:.1f}% complete!")

        slow_factor = 1.0
        for point in plan.joint_trajectory.points:
            point.time_from_start *= slow_factor
            point.velocities = [v / slow_factor for v in point.velocities]
            point.accelerations = [a / (slow_factor**2) for a in point.accelerations]

        self.move_group.execute(plan, wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

        return True


    def reach(self, command): 
        starting_command = self.current_pose

        # first phase : 
        if self.grasp_type != 2 : #medium wrap, lateral pinch : the arm comes back close to the robot
            first_phase_position = (self.positions[0][0], self.positions[starting_command][1], self.positions[starting_command][2])

        else : #power sphere : the arm comes goes up
            first_phase_position = (self.positions[starting_command][0], self.positions[starting_command][1], self.positions[starting_command][2] + 0.2)

        first_phase_orientation = self.orientations[starting_command]
        success = self.reach_cartesian(first_phase_position, first_phase_orientation)
        rospy.sleep(0.5)

        if not success:
            rospy.logerr("Failed to reach the first phase position.")
            return False

        # second phase : 
        if self.grasp_type != 2 : #medium wrap, lateral pinch : the arm moves to the correct y and z position , and the right orientation
            second_phase_position = (self.positions[0][0], self.positions[command][1], self.positions[command][2])
        
        else : #power sphere : the arm moves to the correct x and y position, and the right orientation
            second_phase_position = (self.positions[command][0], self.positions[command][1], self.positions[starting_command][2] + 0.2)

        second_phase_orientation = self.orientations[command]
        success = self.reach_cartesian(second_phase_position, second_phase_orientation)
        rospy.sleep(0.5)

        if not success:
            rospy.logerr("Failed to reach the second phase position.")
            return False
        
        # third phase : the arm moves to the correct position (can be perturbed)
        if self.position_sigma > 0:
            final_position = np.array(self.positions[command]) + np.random.normal(0, self.position_sigma, 3)
        else:
            final_position = self.positions[command]

        if self.orientation_sigma > 0:
            final_orientation = np.array(self.orientations[command]) + np.random.normal(0, self.orientation_sigma, 3)  
        else :
            final_orientation = self.orientations[command]

        success = self.reach_cartesian(final_position, final_orientation)
        rospy.sleep(0.5)

        if not success:
            rospy.logerr("Failed to reach the final position.")
            return False

        rospy.loginfo("Motion complete.")
        self.current_pose = command
        return True

    
    def lift(self):
        lifting_position = (self.positions[self.current_pose][0], self.positions[self.current_pose][1], self.positions[self.current_pose][2] + 0.2)
        lifting_orientation = self.orientations[self.current_pose]
        success = self.reach_cartesian(lifting_position, lifting_orientation)
        self.timing_pub.publish(Int32(1))  # Notify that lifting has started
        rospy.sleep(2.5)
        self.timing_pub.publish(Int32(0))  # Notify that lifting has ended
        rospy.sleep(0.5)
        if not success:
            rospy.logerr("Failed to reach the lifting position.")
            return False
        
        
        back_position = self.positions[self.current_pose]
        back_orientation = self.orientations[self.current_pose]

        if self.grasp_type != 2 : #only go down for medium wrap and lateral pinch 
            success = self.reach_cartesian(back_position, back_orientation)
            rospy.sleep(0.5)
            if not success:
                rospy.logerr("Failed to return to the original position after lifting.")
                return False
        
        rospy.loginfo("Lifting complete.")
        return True



    def sync_callback(self, msg):
        """Callback for incoming messages from Julia."""
        command = int(msg.data)
        if 1 <= command <= 10:
            rospy.loginfo(f"Received command: {command} ")

            if self.current_pose != 0:
                self.lift()

            self.preshape_pub.publish(Int32(self.grasp_type))
            rospy.sleep(0.5)  # wait for the hand to preshape

            if command == 1:
                success = self.reach_cartesian(self.positions[0], self.orientations[0])
            else : 
                success = self.reach(command - 1)

            if success:
                rospy.sleep(0.5)  # small delay before publishing back
                self.sync_pub.publish(Int32(0))
                

        elif command == 0:
            pass
        else:
            rospy.logwarn(f"Received invalid command: {command}")

    def shutdown(self):
        moveit_commander.roscpp_shutdown()
        rospy.loginfo("MoveIt Commander shut down.")


if __name__ == "__main__":

    try:
        controller = UR10eMoveItController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
