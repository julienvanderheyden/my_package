#!/usr/bin/env python3
import rospy
from std_msgs.msg import Int32, String
from geometry_msgs.msg import Point, Quaternion
from math import pi
import numpy as np
import tf.transformations as tf
from my_package.srv import MoveCartesian, MoveCartesianRequest


class GraspOrchestrator:
    def __init__(self):
        rospy.init_node('grasp_orchestrator')
        
        # Publishers
        self.grasp_command_pub = rospy.Publisher('/grasp_command', String, queue_size=10)
        self.preshape_pub = rospy.Publisher('/preshape', Int32, queue_size=10)
        
        # Subscribers
        self.grasp_status_sub = rospy.Subscriber('/grasp_status', Int32, self.grasp_status_callback)
        
        # Service proxies for arm control
        rospy.wait_for_service('/arm/move_cartesian')
        self.move_arm = rospy.ServiceProxy('/arm/move_cartesian', MoveCartesian)
        
        # State variables
        self.grasp_complete = False
        self.grasp_success = False
        self.current_step = 0

        ###################### MODIFY BELOW FOR DIFFERENT TESTS ######################
        
        # Test configuration

        self.grasp_type = 1  # 1: medium wrap, 2: power sphere, 3: lateral pinch
        self.dimensions = [0.01, 0.015, 0.02, 0.0225, 0.025, 0.0275, 0.03, 0.035, 0.04] # cylinders
        # self.dimensions = [0.01, 0.015, 0.015, 0.015, 0.02, 0.02, 0.02, 0.025, 0.03] # prisms 
        self.parameters = (0.8*np.array(self.dimensions)).tolist()    
        # self.parameters = [0.006, 0.012, 0.015, 0.018, 0.02, 0.025, 0.0275, 0.03, 0.035] # cylinders
        # self.parameters = [0.006, 0.012, 0.012, 0.015, 0.017, 0.02,  0.02, 0.026, 0.032 ] # prisms


        # self.grasp_type = 2
        # self.dimensions =  [0.019, 0.019, 0.03, 0.035, 0.04] # spheres
        # self.dimensions = [0.019, 0.019, 0.03, 0.035, 0.04] # cubes
        # self.parameters = 0.8*self.dimensions
        # self.parameters = [0.016, 0.019, 0.024, 0.028, 0.032] # spheres
        # self.parameters = [0.016, 0.019, 0.024, 0.028, 0.032] # cubes
        
        # self.grasp_type = 3 
        # self.dimensions = [[0.025, 0.001], [0.025, 0.0075], [0.025, 0.015],[0.0375, 0.001], [0.0375, 0.0075], [0.0375, 0.015],[0.05, 0.001], [0.05, 0.0075], [0.05, 0.015]] # boxes and disks
        # self.parameters = 0.8*self.dimensions
        # self.parameters = [[0.025, 0.001], [0.025, 0.006], [0.025, 0.013],[0.0375, 0.001], [0.0375, 0.006], [0.0375, 0.013],[0.05, 0.001], [0.05, 0.006], [0.05, 0.013]] # boxes and disks

        # Dimension noise parameters
        self.dimension_noise_is_fixed = False
        self.fixed_dimension_noise = 0.0
        self.std_dimension_noise = 0.1

        self.noisy_parameters = []

        for grasp_params in self.parameters:

            # Force params to be a 1-D numpy array
            params = np.atleast_1d(np.array(grasp_params, dtype=float))

            # Fixed or random noise
            if self.dimension_noise_is_fixed:
                noise = np.full_like(params, self.fixed_dimension_noise, dtype=float)
            else:
                noise = np.random.normal(0, self.std_dimension_noise, size=params.size)

            adjusted_params = (1 + noise) * params
            self.noisy_parameters.append(adjusted_params)

        rospy.loginfo(f"Parameters : {self.parameters}, Noisy Parameters: {self.noisy_parameters}")
        
        # Position noise parameters
        self.position_noise_enabled = True
        self.translation_noise_offset = [0.0, 0.0, 0.0] # [x, y, z] in meters
        self.orientation_noise_offset = [0.0, 0.0, 10.0] # [yaw, pitch, roll] in degrees

        self.orientation_noise_offset = [angle * (pi / 180) for angle in self.orientation_noise_offset]  # Convert to radians
        
        # Reference positions
        if self.grasp_type == 2 : #power sphere
            self.reference_positions = [
                [0.90, 0.173, 1.07], #home position
                [1.268, 0.738, 0.866], #left most position
                [1.268, 0.499, 0.866],
                [1.268, 0.258, 0.866],
                [1.268, 0.023, 0.866],
                [1.268, -0.219, 0.866]] # right most position
        else :
            self.reference_positions = [
                [0.90, 0.173, 1.07],     # 0: home
                [1.268, 0.738, 0.866],   # 1: leftmost
                [1.268, 0.618, 0.866],   # 2
                [1.268, 0.499, 0.866],   # 3
                [1.268, 0.379, 0.866],   # 4
                [1.268, 0.258, 0.866],   # 5
                [1.268, 0.139, 0.866],   # 6
                [1.268, 0.023, 0.866],   # 7
                [1.268, -0.098, 0.866],  # 8
                [1.268, -0.219, 0.866],  # 9: rightmost
            ]
            
        # Compute all positions
        self.positions = self._compute_all_positions()
        self.orientations = self._compute_all_orientations()
        
        rospy.sleep(1.0)
        rospy.loginfo("Grasp Orchestrator ready")

    def _compute_all_positions(self):
        """Compute palm positions for all objects"""
        positions = [self.reference_positions[0]]  # Home position
        
        for i, dim in enumerate(self.dimensions):
            palm_pos = self.compute_palm_position(
                self.reference_positions[i + 1], 
                dim, 
                self.noisy_parameters[i],
                i + 1
            )
            positions.append(palm_pos)
        
        return positions

    def _compute_all_orientations(self):
        """Compute orientations (as quaternions) for all positions."""
        if self.grasp_type != 2:  # medium wrap, lateral pinch
            euler = [pi, -pi/2, 0]  # roll, pitch, yaw
        else:  # power sphere
            euler = [pi/2, 0, pi/2]  # roll, pitch, yaw
        
        # Convert to quaternion
        quat = tf.quaternion_from_euler(*euler)
        return [quat.tolist()] * len(self.positions)

    def compute_palm_position(self, ref_position, dim, param, i):
        """Compute palm position based on grasp type and parameters."""
        if self.grasp_type == 1:  # medium wrap
            radius = dim
            alpha = 1.31
            z = ref_position[2] + 0.09
            
            if param[0] >= 0.015:
                y = ref_position[1] - radius - 0.02
                x = ref_position[0] - (radius + 0.01) * (np.cos(alpha/2) / np.sin(alpha/2)) - 0.04
            else:
                y = ref_position[1] - radius - 0.015
                x = ref_position[0] - 0.075
            
            return [x, y, z]
        
        elif self.grasp_type == 2:  # power sphere
            radius = dim
            stand_height = 0.17
            depth_ratio = 4/3
            
            if param > 0.025:
                z = ref_position[2] + stand_height + depth_ratio * radius
            else:
                z = ref_position[2] + stand_height + (1 - depth_ratio) * radius + 0.11
            
            y = ref_position[1] - 0.03
            x = ref_position[0] - 0.083
            
            return [x, y, z]
        
        else:  # lateral pinch
            if i <= 5:
                support_height = 0.19
            else:
                support_height = 0.225
            
            palm_knuckle_dist = 0.033
            finger_width = 0.018
            z = ref_position[2] + support_height - palm_knuckle_dist - finger_width/2
            y = ref_position[1] - 0.03
            x = ref_position[0] - 0.125
            
            return [x, y, z]

    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles (roll, pitch, yaw) to quaternion [x, y, z, w]."""
        quat = tf.quaternion_from_euler(roll, pitch, yaw)
        return quat.tolist()

    def apply_orientation_noise(self, base_quat, noise_rpy):
        """
        Apply orientation noise (specified in roll, pitch, yaw) to a base quaternion.
        
        Args:
            base_quat: Base orientation as quaternion [x, y, z, w]
            noise_rpy: Noise in Euler angles [roll, pitch, yaw] in radians
        
        Returns:
            Perturbed quaternion [x, y, z, w]
        """
        # Convert noise from Euler to quaternion
        noise_quat = tf.quaternion_from_euler(*noise_rpy)
        
        # Multiply quaternions: result = base * noise
        result_quat = tf.quaternion_multiply(base_quat, noise_quat)
        
        return result_quat.tolist()

    def move_to_pose(self, position, orientation_quat, speed_factor=1.0):
        """Move arm to specified pose."""
        try:
            req = MoveCartesianRequest()
            req.position = Point(*position)
            req.orientation = Quaternion(*orientation_quat)
            req.eef_step = 0.005
            req.speed_factor = speed_factor
            
            response = self.move_arm(req)
            
            if not response.success:
                rospy.logerr(f"Failed to move arm: {response.message}")
            
            return response.success
        
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False

    def execute_reach_sequence(self, target_idx):
        """Execute multi-phase reach motion to target position."""
        starting_idx = self.current_step
        
        # Phase 1: Retract
        if self.grasp_type != 2:  # medium wrap, lateral pinch
            phase1_pos = [
                self.positions[0][0],
                self.positions[starting_idx][1],
                self.positions[starting_idx][2]
            ]
        else:  # power sphere
            phase1_pos = [
                self.positions[starting_idx][0],
                self.positions[starting_idx][1],
                self.positions[starting_idx][2] + 0.2
            ]
        
        if not self.move_to_pose(phase1_pos, self.orientations[starting_idx]):
            return False
        rospy.sleep(0.5)
        
        # Phase 2: Move to target y/z or x/y
        if self.grasp_type != 2:
            phase2_pos = [
                self.positions[0][0],
                self.positions[target_idx][1],
                self.positions[target_idx][2]
            ]
        else:
            phase2_pos = [
                self.positions[target_idx][0],
                self.positions[target_idx][1],
                self.positions[starting_idx][2] + 0.2
            ]
        
        if not self.move_to_pose(phase2_pos, self.orientations[target_idx]):
            return False
        rospy.sleep(0.5)
        
        # Phase 3: Final position
        final_pos = self.positions[target_idx]
        final_orient = self.orientations[target_idx]
        if not self.move_to_pose(final_pos, final_orient):
            return False
        rospy.sleep(0.5)

        # Phase 4: Apply additional noise if enabled
        if self.position_noise_enabled:
            noisy_pos = (np.array(final_pos) + np.array(self.translation_noise_offset)).tolist()
            noisy_orient = self.apply_orientation_noise(final_orient, self.orientation_noise_offset)
            
            if not self.move_to_pose(noisy_pos, noisy_orient):
                return False
            rospy.sleep(0.5)
        
        self.current_step = target_idx
        return True

    def execute_lift(self):
        """Execute lifting motion."""
        lift_pos = [
            self.positions[self.current_step][0],
            self.positions[self.current_step][1],
            self.positions[self.current_step][2] + 0.2
        ]
        
        
        if not self.move_to_pose(lift_pos, self.orientations[self.current_step]):
            return False
        
        rospy.sleep(3.0)
        
        # Return to original position (except for power sphere)
        if self.grasp_type != 2:
            if not self.move_to_pose(
                self.positions[self.current_step],
                self.orientations[self.current_step]
            ):
                return False
            rospy.sleep(0.5)
        
        return True

    def execute_grasp(self, params):
        """Execute grasp with given parameters."""
        
        self.grasp_complete = False
        self.grasp_success = False

        # Format command string
        command = f"{self.grasp_type}," + ",".join(map(str, params))

        rospy.loginfo(f"Sending grasp command: {command}")
        self.grasp_command_pub.publish(String(data=command))
        
        # Wait for completion
        timeout = rospy.Time.now() + rospy.Duration(45.0)
        rate = rospy.Rate(10)
        
        while not self.grasp_complete and rospy.Time.now() < timeout:
            rate.sleep()
        
        if not self.grasp_complete:
            rospy.logerr("Grasp execution timed out")
            return False
        
        return self.grasp_success

    def grasp_status_callback(self, msg):
        """Callback for grasp execution status."""
        if msg.data == 0:
            rospy.loginfo("Grasp executed successfully")
            self.grasp_success = True
        else:
            rospy.logerr("Grasp execution failed")
            self.grasp_success = False
        
        self.grasp_complete = True

    def run_sequence(self):
        """Execute full grasp sequence."""
        # Go to home position
        rospy.loginfo("Moving to home position...")
        if not self.move_to_pose(self.positions[0], self.orientations[0]):
            rospy.logerr("Failed to reach home position")
            return False
        
        self.current_step = 0
        rospy.sleep(1.0)
        
        for i, params in enumerate(self.noisy_parameters):
            rospy.loginfo(f"\n=== Executing grasp {i+1}/{len(self.parameters)} ===")
            
            # Preshape hand
            rospy.loginfo("Preshaping hand...")
            self.preshape_pub.publish(Int32(self.grasp_type))
            rospy.sleep(0.5)
            
            # Move to grasp position
            rospy.loginfo(f"Moving to position {i+1}...")
            if not self.execute_reach_sequence(i + 1):
                rospy.logerr(f"Failed to reach position {i+1}")
                return False
            
            # Execute grasp
            rospy.loginfo("Executing grasp...")
            if not self.execute_grasp(params.tolist()):
                rospy.logerr(f"Failed to execute grasp {i+1}")
                return False
            
            # Lift object
            rospy.loginfo("Lifting object...")
            if not self.execute_lift():
                rospy.logerr("Failed to lift")
                return False
            

        # Return home
        rospy.loginfo("Returning to home position...")
        self.preshape_pub.publish(Int32(self.grasp_type))
        rospy.sleep(0.5)
        if self.grasp_type == 1:
            home_waypoint = [self.positions[-1][0] - 0.2, self.positions[-1][1], self.positions[-1][2]]
            self.move_to_pose(home_waypoint, self.orientations[-1])
        if not self.move_to_pose(self.positions[0], self.orientations[0]):
            rospy.logerr("Failed to return home")
            return False
        
        rospy.loginfo("\n=== Sequence complete! ===")
        return True


if __name__ == '__main__':
    try:
        orchestrator = GraspOrchestrator()
        orchestrator.run_sequence()
    except rospy.ROSInterruptException:
        pass