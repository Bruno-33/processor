#!/usr/bin/env python3
import rospy
from lion_brain.msg import rc
from lion_brain.msg import chassis_control
import random

mode = 0
fake_publisher = ""


def rc_msg_receiver(msg):
    global mode
    mode = msg.assist_mode


def fake_msg_processor(data):
    global fake_publisher
    out = chassis_control()
    out.x = 1#random.uniform(-0.4, 0.4)
    out.y = 0#random.uniform(-0.4, 0.4)
    fake_publisher.publish(out)


try:
    rospy.init_node('fake_processor', anonymous=True)
    rospy.Subscriber("/lion_brain/rc", rc, rc_msg_receiver)
    #rospy.Subscriber("/lion_brain/chassis_control1", chassis_control, fake_msg_processor)
    fake_publisher = rospy.Publisher("/lion_brain/chassis_control", chassis_control, queue_size=10)
    #rospy.spin()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        out = chassis_control()
        out.x = 0.7#random.uniform(-0.4, 0.4)
        out.y = 0#random.uniform(-0.4, 0.4)
        fake_publisher.publish(out)
        rate.sleep()
except rospy.ROSInterruptException:
    print("exception")
    pass


