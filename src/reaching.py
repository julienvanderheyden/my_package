#!/usr/bin/env python3
import rospy
from your_package_name.srv import GetStableEstimate, GetCylinderGrasp

def plan_cylinder_grasp():
    rospy.init_node('cylinder_grasp_planner_node')

    # Wait for services and setup proxies outside the try block
    rospy.wait_for_service('/perception/get_stable_estimate')
    rospy.wait_for_service('/grasp_planning/get_cylinder_grasp')

    get_estimate = rospy.ServiceProxy('/perception/get_stable_estimate', GetStableEstimate)
    get_grasp = rospy.ServiceProxy('/grasp_planning/get_cylinder_grasp', GetCylinderGrasp)

    try:
        est = get_estimate()
        if not est.success:
            rospy.logerr(f"Perception failed. Reason: {est.reason}")
            return
        
        elif est.primitive_type != "CYLINDER":
            rospy.logwarn(f"Expected CYLINDER, but got '{est.primitive_type}'")
            return
        else:
            grasp_req = GetCylinderGraspRequest()
            grasp_req.primitive_type = est.primitive_type
            grasp_req.estimate = est.estimate
            grasp_req.cloud = est.cloud
            print(get_grasp(grasp_req))

    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

if __name__ == '__main__':
    plan_cylinder_grasp()
