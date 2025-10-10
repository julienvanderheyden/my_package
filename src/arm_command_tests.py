#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
import geometry_msgs.msg
from math import pi
import argparse
from tf.transformations import quaternion_from_euler
from std_msgs.msg import Int32
from copy import deepcopy


class UR10eMoveItController:
    def __init__(self, grasp_type, group_name="right_arm"):
        # Initialize MoveIt Commander and ROS node
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("ur10e_cartesian_goal", anonymous=True)

        #grasp type 
        self.grasp_type = grasp_type

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

        # Predefined positions (x, y, z) and orientations (roll, pitch, yaw)
        self.positions = [
            [0.724, 0.173, 1.07],
            [1.0, 0.173, 1.07],
            [1.0, 0.0, 1.07],
            [1.0, 0.2, 1.07],
            [1.0, 0.4, 1.07],
        ]
        self.orientations = [
            [1.5*pi, 0, 0],
            [1.5*pi, 0, 0],
            [1.5*pi, 0, 0],
            [1.5*pi, 0, 0],
            [1.5*pi, 0, 0],
        ]

        self.planning_frame = self.move_group.get_planning_frame()
        self.eef_link = self.move_group.get_end_effector_link()

        rospy.loginfo(f"Planning frame: {self.planning_frame}")
        rospy.loginfo(f"End effector link: {self.eef_link}")

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

        rospy.loginfo("Executing Cartesian path...")
        
        slow_factor = 1.0
        for point in plan.joint_trajectory.points:
            point.time_from_start *= slow_factor
            # Scale velocities and accelerations accordingly
            point.velocities = [v / slow_factor for v in point.velocities]
            point.accelerations = [a / (slow_factor**2) for a in point.accelerations]

        self.move_group.execute(plan, wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        rospy.loginfo("Motion complete.")
        return True


    def sync_callback(self, msg):
        """Callback for incoming messages from Julia."""
        command = int(msg.data)
        if 1 <= command <= 5:
            rospy.loginfo(f"Received command: {command} → moving to pose {command}")
            self.preshape_pub.publish(Int32(self.grasp_type))
            rospy.sleep(1)  # wait for the hand to preshape
            position = self.positions[command - 1]
            orientation = self.orientations[command - 1]
            success = self.reach_cartesian(position, orientation)
            if success:
                rospy.sleep(0.5)  # small delay before publishing back
                rospy.loginfo("Publishing 0 to indicate completion.")
                self.sync_pub.publish(Int32(0))
        elif command == 0:
            rospy.loginfo("Received 0 — no action.")
        else:
            rospy.logwarn(f"Received invalid command: {command}")

    def shutdown(self):
        moveit_commander.roscpp_shutdown()
        rospy.loginfo("MoveIt Commander shut down.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arm motion and hand preshape control node")
    parser.add_argument("grasp_type", type=int, choices=[1, 2, 3],
                        help="Grasp type: 1 = medium wrap, 2 = power sphere, 3 = lateral pinch")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])  # Use rospy.myargv to ignore ROS args

    try:
        controller = UR10eMoveItController(args.grasp_type)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
