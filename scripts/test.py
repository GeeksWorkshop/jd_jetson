#! /usr/bin/env python

import rospy

import actionlib
from net.hello import hello
import actionlib_tutorials.msg
from net.video import start
# print('hello ',hello)
# hello()
print(start)
start(1)
