#!/usr/bin/env python

from laser_assembler.srv import AssembleScans2
import rospy
from sensor_msgs.msg import PointCloud2


if __name__ == '__main__':
    rospy.init_node('assembled_scans_timer_publisher')
    pub_cloud = rospy.Publisher('~output', PointCloud2, queue_size=1)
    rate = rospy.get_param('~rate', 40)
    duration = rospy.get_param('~duration', 1.0)
    r = rospy.Rate(rate)

    service_name = '~assemble_scans2'
    rospy.wait_for_service(service_name)

    while not rospy.is_shutdown():
        try:
            assemble_scans2 = rospy.ServiceProxy(service_name, AssembleScans2)
            now = rospy.Time.now()
            res = assemble_scans2(now - rospy.Duration(duration), now)
            pub_cloud.publish(res.cloud)
        except rospy.ServiceException as e:
            rospy.loginfo('Service call failed.\n{}'.format(e))

        r.sleep()
