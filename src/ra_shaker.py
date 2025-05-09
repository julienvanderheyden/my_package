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

    shake_left = [0.25, -0.92, 1.6, -0.72, 1.6, -1.5]  
    shake_right = [-0.25, -0.92, 1.6, -0.72, 1.6, -1.5]

    time_stages = [3.0, 2.0, 1.0]  # seconds between shakes

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

if __name__ == '__main__':
    try:
        send_shake_trajectory()
    except rospy.ROSInterruptException:
        pass
