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

        self.preshape_pub = rospy.Publisher("/preshape", Int32, queue_size=10)

        ### THIS SHOULD BE CHANGED FOR EACH TEST ###
    
        #grasp type 
        self.grasp_type = 1 # medium wrap

        # Predefined positions (x, y, z) and orientations (roll, pitch, yaw)
        self.positions = [
            [0.724, 0.173, 1.07], # 1 : home position
            [1.0, -0.2, 1.07], # 2 : leftmost position
            [1.0, 0.0, 1.07], # 3
            [1.0, 0.2, 1.07], # 4 
            [1.0, 0.4, 1.07], # 5 
            [1.0, 0.6, 1.07], # 6 : leftmost position
        ]
        self.orientations = [
            [1.5*pi, 0, 0],
            [1.5*pi, 0, 0],
            [1.5*pi, 0, 0],
            [1.5*pi, 0, 0],
            [1.5*pi, 0, 0],
            [1.5*pi, 0, 0],
        ]

        self.position_sigma = 0.01
        self.orientation_sigma = 0.0

        ##############################################

        self.planning_frame = self.move_group.get_planning_frame()
        self.eef_link = self.move_group.get_end_effector_link()
        self.current_pose = 0 

        rospy.loginfo(f"Planning frame: {self.planning_frame}")
        rospy.loginfo(f"End effector link: {self.eef_link}")

        self.reach_cartesian(self.positions[0], self.orientations[0]) #going to home position
        rospy.loginfo("UR10eMoveItController initialized and listening for commands...")

    def get_current_pose(self):
        """Return the current end-effector pose (geometry_msgs/Pose)."""
        return self.move_group.get_current_pose(self.eef_link).pose

    def reach_cartesian(self, position, orientation, eef_step=0.005):
        """
        Move the end-effector to a target position using a smooth Cartesian path.

        Args:
            position: (x, y, z) in meters
            orientation: (roll, pitch, yaw) in radians
            eef_step: distance between waypoints in meters
        """

        current_pose = deepcopy(self.get_current_pose())
        target_pose = deepcopy(current_pose)

        # Set target position
        target_pose.position.x = position[0]
        target_pose.position.y = position[1]
        target_pose.position.z = position[2]

        # Set orientation
        roll, pitch, yaw = orientation
        q = quaternion_from_euler(roll, pitch, yaw)
        target_pose.orientation.x = q[0]
        target_pose.orientation.y = q[1]
        target_pose.orientation.z = q[2]
        target_pose.orientation.w = q[3]

        # Generate waypoints
        waypoints = [target_pose]

        # Compute Cartesian path
        (plan, fraction) = self.move_group.compute_cartesian_path(
            waypoints,
            eef_step=eef_step,  # e.g., 1 cm per step
            jump_threshold=0.0  # prevents IK jumps
        )

        if fraction < 1.0:
            rospy.logwarn(f"Cartesian path only {fraction*100:.1f}% complete!")
        
        slow_factor = 1.0
        for point in plan.joint_trajectory.points:
            point.time_from_start *= slow_factor
            # Scale velocities and accelerations accordingly
            point.velocities = [v / slow_factor for v in point.velocities]
            point.accelerations = [a / (slow_factor**2) for a in point.accelerations]

        self.move_group.execute(plan, wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        
        return True

    def reach(self, command): 
        starting_command = self.current_pose

        # first phase : the arm comes back close to the robot
        first_phase_position = (self.positions[0][0], self.positions[starting_command][1], self.positions[starting_command][2])
        first_phase_orientation = self.orientations[starting_command]
        success = self.reach_cartesian(first_phase_position, first_phase_orientation)
        rospy.sleep(0.5)

        if not success:
            rospy.logerr("Failed to reach the first phase position.")
            return False

        # second phase : the arm moves to the correct y and z position, and the right orientation
        second_phase_position = (self.positions[0][0], self.positions[command][1], self.positions[command][2])
        second_phase_orientation = self.orientations[command]
        success = self.reach_cartesian(second_phase_position, second_phase_orientation)
        rospy.sleep(0.5)

        if not success:
            rospy.logerr("Failed to reach the second phase position.")
            return False
        
        # third phase : the arm moves to the correct x position (can be perturbed)
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
        rospy.sleep(3.0)
        if not success:
            rospy.logerr("Failed to reach the lifting position.")
            return False
        
        back_position = self.positions[self.current_pose]
        back_orientation = self.orientations[self.current_pose]
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
        if 1 <= command <= 6:
            rospy.loginfo(f"Received command: {command} ")

            if self.current_pose != 0:
                self.lift()
                rospy.sleep(0.5)

            self.preshape_pub.publish(Int32(self.grasp_type))
            rospy.sleep(1)  # wait for the hand to preshape
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
