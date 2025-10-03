#!/usr/bin/env python3

import rospy
import argparse
from std_msgs.msg import Empty
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class ShakeTestNode:
    def __init__(self, grasp_type):
        rospy.init_node('shake_test')

        self.joint_names = [
            'ra_shoulder_pan_joint', 'ra_shoulder_lift_joint', 'ra_elbow_joint',
            'ra_wrist_1_joint', 'ra_wrist_2_joint', 'ra_wrist_3_joint'
        ]

        self.pregrasp_position = [-0.4528, -0.997, 2.278, 1.2558, 1.1071, -1.564]

        # Define grasping types
        grasp_types = {
            1: [-0.28, -0.74, 1.66, -0.89, 1.27, -1.55],     # Medium wrap
            2: [-0.308, -0.875, 1.549, -0.697, 1.2, -3.146],  # Power sphere
            3: [-0.363, -0.805, 1.91, -1.177, 1.1, -1.535],   # Lateral pinch
        }

        if grasp_type not in grasp_types:
            raise ValueError(f"Invalid grasp_type: {grasp_type}. Choose 1 (medium wrap), 2 (power sphere), or 3 (lateral pinch)")

        self.grasping_position = grasp_types[grasp_type]

        self.lifting_position = [-0.285, -1.048, 1.575, -0.503, 1.27, -1.55]
        self.shake_left = [-0.258, -1.048, 1.575, -0.503, 1.27 + 0.3, -1.55]
        self.shake_right = [-0.258, -1.048, 1.575, -0.503, 1.27 - 0.3, -1.55]
        self.shake_up = [-0.258, -1.048, 1.575, -0.503 + 0.3, 1.27, -1.55]
        self.shake_down = [-0.258, -1.048, 1.575, -0.503 - 0.3, 1.27, -1.55]

        self.time_stages = [1.0, 0.5, 0.25]

        self.pub = rospy.Publisher('/ra_trajectory_controller/command', JointTrajectory, queue_size=10)

        # Subscribers for triggering actions
        rospy.Subscriber('/start_grasping', Empty, self.callback_grasping)
        rospy.Subscriber('/start_testing_x', Empty, self.callback_testing_x)
        rospy.Subscriber('/start_testing_z', Empty, self.callback_testing_z)

        rospy.loginfo("Waiting for controller publisher...")
        while self.pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("Connected to controller.")

        self.send_empty_trajectory()
        self.send_trajectory(self.lifting_position, 2.0)
        self.send_trajectory(self.grasping_position, 2.0)
        rospy.loginfo("Moved to grasp position.")

    def send_empty_trajectory(self):
        empty_traj = JointTrajectory()
        empty_traj.joint_names = self.joint_names
        empty_traj.header.stamp = rospy.Time.now() + rospy.Duration(0.1)
        self.pub.publish(empty_traj)
        rospy.sleep(0.5)
        rospy.loginfo("Initialized empty trajectory.")

    def send_trajectory(self, position, duration):
        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = position
        point.time_from_start = rospy.Duration(duration)
        traj.points.append(point)
        traj.header.stamp = rospy.Time.now() + rospy.Duration(0.1)
        self.pub.publish(traj)
        rospy.sleep(duration)

    def callback_grasping(self, msg):
        rospy.loginfo("Received start_grasping signal.")
        self.send_trajectory(self.grasping_position, 2.0)
        rospy.loginfo("Grasping movement complete.")

    def callback_testing_x(self, msg):
        rospy.loginfo("Received start_testing_x signal.")
        self.run_test_sequence(direction='x')

    def callback_testing_z(self, msg):
        rospy.loginfo("Received start_testing_z signal.")
        self.run_test_sequence(direction='z')

    def run_test_sequence(self, direction='x'):
        rospy.loginfo("Lifting object...")
        self.send_trajectory(self.lifting_position, 2.0)
        rospy.loginfo("Waiting 5 seconds...")
        rospy.sleep(5.0)

        if direction == 'x':
            pos1, pos2 = self.shake_left, self.shake_right
        elif direction == 'z':
            pos1, pos2 = self.shake_up, self.shake_down
        else:
            rospy.logwarn(f"Unknown direction '{direction}', skipping test.")
            return

        for stage_index, time_step in enumerate(self.time_stages):
            rospy.loginfo(f"Shake stage {stage_index + 1} on {direction}-axis with time_step = {time_step:.2f}s")
            traj = JointTrajectory()
            traj.joint_names = self.joint_names
            traj.points = []

            current_time = 0.0
            for _ in range(3):
                p1 = JointTrajectoryPoint()
                p1.positions = pos1
                current_time += time_step
                p1.time_from_start = rospy.Duration(current_time)
                traj.points.append(p1)

                p2 = JointTrajectoryPoint()
                p2.positions = pos2
                current_time += time_step
                p2.time_from_start = rospy.Duration(current_time)
                traj.points.append(p2)

            traj.header.stamp = rospy.Time.now() + rospy.Duration(0.1)
            self.pub.publish(traj)
            rospy.sleep(current_time + 1.0)

        rospy.loginfo("Shaking complete. Returning to grasping position.")
        self.send_trajectory(self.grasping_position, 2.0)

if __name__ == '__main__':
    import sys

    parser = argparse.ArgumentParser(description="Shake test node for robotic grasp evaluation.")
    parser.add_argument("grasp_type", type=int, choices=[1, 2, 3],
                        help="Grasp type: 1 = medium wrap, 2 = power sphere, 3 = lateral pinch")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])  # Use rospy.myargv to ignore ROS args

    try:
        node = ShakeTestNode(args.grasp_type)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
