#!/usr/bin/env python3

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def send_shake_trajectory():
    rospy.init_node('shake_test')

    pub = rospy.Publisher('/ra_trajectory_controller/command', JointTrajectory, queue_size=10)

    # # Wait until publisher is connected
    rospy.loginfo("Waiting for controller publisher...")
    while pub.get_num_connections() == 0 and not rospy.is_shutdown():
        rospy.sleep(0.1)
    rospy.loginfo("Connected to controller!")

    joint_names = ['ra_shoulder_pan_joint', 'ra_shoulder_lift_joint', 'ra_elbow_joint',
                    'ra_wrist_1_joint', 'ra_wrist_2_joint', 'ra_wrist_3_joint']

    grasping_position = [-0.07, -0.93, 1.764, -0.651, 1.771, -0.898]
    lifting_position = [0.0, -0.92, 1.6, -0.72, 1.6, -1.5]
    shake_left = [0.2, -0.92, 1.6, -0.72, 1.6, -1.5]  
    shake_right = [-0.2, -0.92, 1.6, -0.72, 1.6, -1.5]

    time_stages = [3.0, 1.0, 0.5]  # seconds between shakes

    # Send empty trajectory to initialize
    empty_traj = JointTrajectory()
    empty_traj.joint_names = joint_names
    empty_traj.header.stamp = rospy.Time.now() + rospy.Duration(0.1)

    pub.publish(empty_traj)
    rospy.sleep(0.5)  # give the controller time to react
    rospy.loginfo("Initialized empty trajectory.")

    rospy.loginfo(f"Starting lifting test")
    traj = JointTrajectory()
    traj.joint_names = joint_names
    lifting_point = JointTrajectoryPoint()
    lifting_point.positions = lifting_position
    lifting_point.time_from_start = rospy.Duration(2.0)
    traj.points.append(lifting_point)
    traj.header.stamp = rospy.Time.now() + rospy.Duration(0.1)
    pub.publish(traj)
    rospy.sleep(2.0)  # give the controller time to lift
    rospy.loginfo("Lifting test complete.")

    for stage_index, time_step in enumerate(time_stages):
        rospy.loginfo(f"Starting shake stage {stage_index + 1} with time_step = {time_step:.2f}s")

        # Create trajectory message
        traj = JointTrajectory()
        traj.joint_names = joint_names
        traj.points = []

        current_time = 0.0  

        for i in range(3):  # Repeat 3 times per stage
            # Shake left
            p1 = JointTrajectoryPoint()
            p1.positions = shake_left
            current_time += time_step
            p1.time_from_start = rospy.Duration(current_time)
            traj.points.append(p1)

            # Shake right
            p2 = JointTrajectoryPoint()
            p2.positions = shake_right
            current_time += time_step
            p2.time_from_start = rospy.Duration(current_time)
            traj.points.append(p2)
        
        traj.header.stamp = rospy.Time.now() + rospy.Duration(0.1)
        pub.publish(traj)

        rospy.sleep(current_time + 1.0)

    rospy.loginfo("Shaking test complete.")

    # Send the hand to the grasping position
    rospy.loginfo("Sending hand to grasping position.")
    traj = JointTrajectory()
    traj.joint_names = joint_names
    grasping_point = JointTrajectoryPoint()
    grasping_point.positions = grasping_position
    grasping_point.time_from_start = rospy.Duration(2.0)
    traj.points.append(grasping_point)
    traj.header.stamp = rospy.Time.now() + rospy.Duration(0.1)
    pub.publish(traj)
    rospy.sleep(2.0)  # give the controller time to grasp
    rospy.loginfo("Grasping test complete.")

if __name__ == '__main__':
    try:
        send_shake_trajectory()
    except rospy.ROSInterruptException:
        pass
