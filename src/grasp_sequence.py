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
        
        rospy.wait_for_service('/arm/move_cartesian')
        self.move_arm = rospy.ServiceProxy('/arm/move_cartesian', MoveCartesian)
        
        self.grasp_complete = False
        self.grasp_success = False
        self.current_step = 0
        self.last_noisy_position = []
        self.last_noisy_orientation = []

        self.reference_positions = [
            [0.90, 0.173, 1.07],     # 0: home
            [1.35, 0.68, 0.74],   # 1: leftmost
            [1.35, 0.56, 0.74],   # 2
            [1.35, 0.44, 0.74],   # 3
            [1.35, 0.325, 0.74],   # 4
            [1.35, 0.205, 0.74],   # 5
            [1.35, 0.085, 0.74],   # 6
            [1.35, -0.035, 0.74],  # 7
            [1.35, -0.15, 0.74],  # 8
            [1.35, -0.27, 0.74],  # 9
            [1.35, -0.39, 0.74],  # 10: rightmost
        ]
    
        ###################### MODIFY BELOW FOR DIFFERENT TESTS ######################
        
        self.grasp_type_configs = {
            1: {  # medium wrap
                'dimension_noise_is_fixed': True,
                'fixed_dimension_noise': 0.0,
                'std_dimension_noise': 0.0,
                'position_noise_enabled': False,
                'translation_noise_offset': [-0.005, 0.0, 0.0],
                'orientation_noise_offset': [0.0, 0.0, 0.0]  # [yaw, pitch, roll] in degrees
            },
            2: {  # power sphere
                'dimension_noise_is_fixed': True,
                'fixed_dimension_noise': 0.0,
                'std_dimension_noise': 0.0,
                'position_noise_enabled': False,
                'translation_noise_offset': [-0.005, 0.0, 0.0],
                'orientation_noise_offset': [0.0, 0.0, 0.0]
            },
            3: {  # lateral pinch
                'dimension_noise_is_fixed': True,
                'fixed_dimension_noise': 0.0,
                'std_dimension_noise': 0.0,
                'position_noise_enabled': True,
                'translation_noise_offset': [-0.005, 0.0, 0.0],
                'orientation_noise_offset': [0.0, 0.0, 0.0]
            }
        }
        
        self.grasp_sequence = [
            {'position_idx': 1, 'grasp_type': 2, 'dimension': 0.04, 'parameters': 0.0375},
            #{'position_idx': 2, 'grasp_type': 1, 'dimension': 0.015, 'parameters': 0.035},
            {'position_idx': 3, 'grasp_type': 2, 'dimension': 0.0425, 'parameters': 0.0375},
            #{'position_idx': 4, 'grasp_type': 1, 'dimension': 0.0225, 'parameters': 0.035},
            {'position_idx': 5, 'grasp_type': 2, 'dimension': 0.045, 'parameters': 0.0375},
            #{'position_idx': 6, 'grasp_type': 1, 'dimension': 0.0275, 'parameters': 0.035},
            #{'position_idx': 7, 'grasp_type': 2, 'dimension': 0.035, 'parameters': 0.045},
            #{'position_idx': 8, 'grasp_type': 1, 'dimension': 0.035, 'parameters': 0.035},
            #{'position_idx': 9, 'grasp_type': 2, 'dimension': 0.0375, 'parameters': 0.045},
        ]

        ###################### END OF CONFIGURATION ######################
        
        self.processed_grasps = []
        self._process_grasp_sequence()
        
        self.positions = self._compute_all_positions()
        self.orientations = self._compute_all_orientations()
        
        rospy.sleep(1.0)
        rospy.loginfo("Multi-Grasp Orchestrator ready")
        rospy.loginfo(f"Total grasps to execute: {len(self.processed_grasps)}")

    def _process_grasp_sequence(self):
        """Process the grasp sequence and compute noisy parameters for each grasp."""
        for grasp_spec in self.grasp_sequence:
            grasp_type = grasp_spec['grasp_type']
            dimension = grasp_spec['dimension']
            parameters = grasp_spec['parameters']
            position_idx = grasp_spec['position_idx']
            
            # Get noise configuration for this grasp type
            config = self.grasp_type_configs[grasp_type]
            
            # Convert orientation noise to radians
            orientation_noise_rad = [
                angle * (pi / 180) for angle in config['orientation_noise_offset']
            ]
            
            # Compute noisy parameters
            params = np.atleast_1d(np.array(parameters, dtype=float))
            
            if config['dimension_noise_is_fixed']:
                noise = np.full_like(params, config['fixed_dimension_noise'], dtype=float)
            else:
                noise = np.random.normal(0, config['std_dimension_noise'], size=params.size)
            
            noisy_params = (1 + noise) * params
            
            # Store processed grasp info
            self.processed_grasps.append({
                'grasp_type': grasp_type,
                'dimension': dimension,
                'parameters': parameters,
                'noisy_parameters': noisy_params,
                'position_idx': position_idx,
                'position_noise_enabled': config['position_noise_enabled'],
                'translation_noise_offset': config['translation_noise_offset'],
                'orientation_noise_offset': orientation_noise_rad
            })
        
        rospy.loginfo(f"Processed {len(self.processed_grasps)} grasps")
        for i, g in enumerate(self.processed_grasps):
            grasp_type_names = {1: 'medium wrap', 2: 'power sphere', 3: 'lateral pinch'}
            rospy.loginfo(f"  Grasp {i+1}: Position {g['position_idx']}, "
                         f"Type {grasp_type_names[g['grasp_type']]}, "
                         f"Params {g['noisy_parameters'].tolist()}")

    def _get_reference_positions(self, grasp_type):
        """Get reference positions based on grasp type."""
        # All grasp types now use the same reference positions
        return self.reference_positions

    def _compute_all_positions(self):
        """Compute palm positions for all objects."""
        positions = [self.reference_positions[0]]  # Home position
        
        for grasp_info in self.processed_grasps:
            grasp_type = grasp_info['grasp_type']
            position_idx = grasp_info['position_idx']
            
            palm_pos = self.compute_palm_position(
                self.reference_positions[position_idx],
                grasp_info['dimension'],
                grasp_info['noisy_parameters'],
                grasp_type,
                position_idx
            )
            positions.append(palm_pos)
        
        return positions

    def _compute_all_orientations(self):
        """Compute orientations (as quaternions) for all positions."""
        orientations = []
        
        # Home orientation
        home_euler = [pi, -pi/2, 0]
        home_quat = tf.quaternion_from_euler(*home_euler)
        orientations.append(home_quat.tolist())
        
        # Compute orientation for each grasp
        for grasp_info in self.processed_grasps:
            if grasp_info['grasp_type'] != 2:  # medium wrap, lateral pinch
                euler = [pi, -pi/2, 0]
            else:  # power sphere
                euler = [pi, -pi/2, 0]
                #euler = [pi/2, 0, pi/2]
            
            quat = tf.quaternion_from_euler(*euler)
            orientations.append(quat.tolist())
        
        return orientations

    def compute_palm_position(self, ref_position, dim, param, grasp_type, ref_idx):
        """Compute palm position based on grasp type and parameters."""
        if grasp_type == 1:  # medium wrap
            radius = dim
            alpha = 1.31
            z = ref_position[2] + 0.09
            
            if radius >= 0.015:
                y = ref_position[1] - radius - 0.003
                x = ref_position[0] - (radius) * (np.cos(alpha/2) / np.sin(alpha/2)) - 0.005
            else:
                y = ref_position[1] - radius
                x = ref_position[0] - 0.025
            
            return [x, y, z]
        
        elif grasp_type == 2:  # power sphere
            radius = dim
            stand_height = 0.17
            depth_ratio = 4/3
            
            # remove power sphere parametric positioning for structured experiments 
            # if param[0] > 0.025:
            #     z = ref_position[2] + stand_height + depth_ratio * radius
            # else:
            #     z = ref_position[2] + stand_height + (1 - depth_ratio) * radius + 0.11

            # z = ref_position[2] + stand_height + depth_ratio * radius
            
            # y = ref_position[1] - 0.03
            # x = ref_position[0] - 0.083

            z = ref_position[2] + stand_height + (1-depth_ratio) *radius
            palm_with = 0.022
            y = ref_position[1] - radius - palm_with/2 + 0.01
            x = ref_position[0] - 0.033
            
            return [x, y, z]
        
        else:  # lateral pinch
            #
            # if ref_idx <= 5:
            #     support_height = 0.19
            # else:
            #     support_height = 0.22
            support_height = 0.19
            
            palm_knuckle_dist = 0.033
            finger_width = 0.018
            z = ref_position[2] + support_height - palm_knuckle_dist - finger_width/2
            y = ref_position[1] - 0.03
            x = ref_position[0] - 0.125
            
            return [x, y, z]

    def apply_orientation_noise(self, base_quat, noise_rpy):
        """Apply orientation noise to a base quaternion."""
        noise_quat = tf.quaternion_from_euler(*noise_rpy)
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

    def execute_reach_sequence(self, target_idx, grasp_type, position_noise_enabled, 
                               translation_offset, orientation_offset):
        """Execute multi-phase reach motion to target position."""
        starting_idx = self.current_step
        
        # Phase 1: Retract
        if grasp_type != 2:  # medium wrap, lateral pinch
            phase1_pos = [
                self.positions[0][0],
                self.positions[starting_idx][1],
                self.positions[starting_idx][2]
            ]
        else:  # power sphere
            # phase1_pos = [
            #     self.positions[starting_idx][0],
            #     self.positions[starting_idx][1],
            #     self.positions[starting_idx][2] + 0.2
            # ]
            phase1_pos = [
                self.positions[0][0],
                self.positions[starting_idx][1],
                self.positions[starting_idx][2]
            ]
        
        if not self.move_to_pose(phase1_pos, self.orientations[starting_idx]):
            return False
        rospy.sleep(0.5)
        
        # Phase 2: Move to target y/z or x/y
        if grasp_type != 2:
            phase2_pos = [
                self.positions[0][0],
                self.positions[target_idx][1],
                self.positions[target_idx][2]
            ]
        else:
            # phase2_pos = [
            #     self.positions[target_idx][0],
            #     self.positions[target_idx][1],
            #     self.positions[starting_idx][2] + 0.2
            # ]
            phase2_pos = [
                self.positions[0][0],
                self.positions[target_idx][1],
                self.positions[target_idx][2]
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
        if position_noise_enabled:
            self.last_noisy_position = (np.array(final_pos) + np.array(translation_offset)).tolist()
            self.last_noisy_orientation = self.apply_orientation_noise(final_orient, orientation_offset)
            
            if not self.move_to_pose(self.last_noisy_position, self.last_noisy_orientation):
                return False
            rospy.sleep(0.5)
        
        self.current_step = target_idx
        return True

    def execute_lift(self, grasp_type, position_noise_enabled):
        """Execute lifting motion."""
        lift_pos = [
            self.positions[self.current_step][0],
            self.positions[self.current_step][1],
            self.positions[self.current_step][2] + 0.2
        ]
        
        lift_orient = self.orientations[self.current_step]

        if not self.move_to_pose(lift_pos, lift_orient):
            return False
        
        rospy.sleep(3.0)
        
        # Return to original position (except for power sphere)
        if grasp_type != 2:
            if position_noise_enabled:
                if not self.move_to_pose(
                    self.last_noisy_position,
                    self.last_noisy_orientation 
                ):
                    return False
            else:
                if not self.move_to_pose(
                    self.positions[self.current_step],
                    self.orientations[self.current_step]
                ):
                    return False
                
            rospy.sleep(0.5)
        
        return True

    def execute_grasp(self, grasp_type, params):
        """Execute grasp with given parameters."""
        self.grasp_complete = False
        self.grasp_success = False

        # Format command string
        command = f"{grasp_type}," + ",".join(map(str, params))

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
        
        for i, grasp_info in enumerate(self.processed_grasps):
            grasp_type = grasp_info['grasp_type']
            params = grasp_info['noisy_parameters']
            target_idx = i + 1  # Offset by 1 because positions[0] is home
            
            grasp_type_names = {1: 'medium wrap', 2: 'power sphere', 3: 'lateral pinch'}
            
            rospy.loginfo(f"\n=== Executing grasp {i+1}/{len(self.processed_grasps)} ===")
            rospy.loginfo(f"Position: {grasp_info['position_idx']}")
            rospy.loginfo(f"Type: {grasp_type_names.get(grasp_type, 'unknown')}")
            rospy.loginfo(f"Parameters: {params.tolist()}")
            
            # Preshape hand
            rospy.loginfo("Preshaping hand...")
            self.preshape_pub.publish(Int32(grasp_type))
            rospy.sleep(0.5)
            
            # Move to grasp position
            rospy.loginfo(f"Moving to position {grasp_info['position_idx']}...")
            if not self.execute_reach_sequence(
                target_idx,
                grasp_type,
                grasp_info['position_noise_enabled'],
                grasp_info['translation_noise_offset'],
                grasp_info['orientation_noise_offset']
            ):
                rospy.logerr(f"Failed to reach position {grasp_info['position_idx']}")
                return False
            
            # Execute grasp
            rospy.loginfo("Executing grasp...")
            if not self.execute_grasp(grasp_type, params.tolist()):
                rospy.logerr(f"Failed to execute grasp {i+1}")
                return False
            
            # Lift object
            rospy.loginfo("Lifting object...")
            if not self.execute_lift(grasp_type, grasp_info['position_noise_enabled']):
                rospy.logerr("Failed to lift")
                return False
            
            self.preshape_pub.publish(Int32(grasp_type))

        # Return home
        rospy.loginfo("Returning to home position...")
        last_grasp_type = self.processed_grasps[-1]['grasp_type']
        self.preshape_pub.publish(Int32(last_grasp_type))
        rospy.sleep(0.5)
        
        if last_grasp_type == 1:
            home_waypoint = [
                self.positions[-1][0] - 0.2,
                self.positions[-1][1],
                self.positions[-1][2]
            ]
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