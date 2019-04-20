#!/usr/bin/env python3
import rospy
import pickle
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from opti_msgs.msg import Odom
from lion_brain.msg import rc
from lion_brain.msg import chassis_control
from plugins import matrix

import numpy as np
import argparse
import torch
import time
import queue
import copy
import configparser

########global variables############
totalRigidbodys = 4 #the id of the goal is 3
robotID = 0
rigidBody_q = [queue.LifoQueue(5) for _ in range(totalRigidbodys-1)]
publisher = ""  ##publish the control command
publisher1 = "" ##publish the debug messages
is_stop = True     ##run the algorithm or not
actor_critic,ob_rms,state,mean,mask=0,0,0,0,0 ##parameters of algorithm

max_velocity = 0
min_velocity = 0
dir_of_model = ""
goal = [0,0]
goal_error = 0

#######read the configuration file and set the parameters######
def configuration(dir):
	global robotID,max_velocity,dir_of_model,min_velocity,goal_error,goal
	conf = configparser.ConfigParser()
	conf.read(dir)
	robotID = int(conf.get('robot','robotID'))
	goal_str = conf.get('robot','goal')
	goal = [float(goal_str[1:goal_str.find(',')]),float(goal_str[goal_str.find(',')+1:len(goal_str)-1])]
	max_velocity = float(conf.get('robot','max_velocity'))
	min_velocity = float(conf.get('robot','min_velocity'))
	goal_error = float(conf.get('robot','goal_error'))
	dir_of_model = conf.get('model','dir')
	print("robotID: ",robotID)
	print("goal: ",goal)
	print("goal_error: ",goal_error)
	print("max_velocity: ",max_velocity)
	print("min_velocity: ",min_velocity)
	print("dir_of_model: ",dir_of_model)

########update the velocity and position of rigidbodys in geodetic coordinate system######
def update_info():
	rigidBody = [Odom()]*totalRigidbodys
	uav_pos = np.array([0.]*(totalRigidbodys-1)*2)
	uav_v = np.array([0.]*(totalRigidbodys-1)*2)
	for i in range(3):
		rigidBody[i] = rigidBody_q[i].get()
	for i in range(totalRigidbodys-1):
		uav_pos[i*2] = rigidBody[i].position.x
		uav_pos[i*2+1] = rigidBody[i].position.y
		uav_v[i*2] = rigidBody[i].linear.x
		uav_v[i*2+1] = rigidBody[i].linear.y
	if np.linalg.norm(uav_v[robotID*2:robotID*2+2]) > max_velocity:
		uav_v[robotID*2:robotID*2+2] = uav_v[robotID*2:robotID*2+2]/np.linalg.norm(uav_v[robotID*2:robotID*2+2])*max_velocity
	if np.linalg.norm(np.array(uav_v[robotID*2],uav_v[robotID*2+1])) < min_velocity:#opti error
		uav_v[robotID*2],uav_v[robotID*2+1] = 0,0
	return rigidBody,uav_pos,uav_v

########get vector in goal coordinate system########
def get_vec():
	rigidBody,uav_pos,uav_v = update_info()
	rigidBody[totalRigidbodys-1].position.x = goal[0]
	rigidBody[totalRigidbodys-1].position.y = goal[1]
	vec = list()
	if np.linalg.norm(uav_v[robotID*2:robotID*2+2]) == 0:
		vec.append(np.array([1,0]))
	else:
		vec.append(uav_v[robotID*2:robotID*2+2]/np.linalg.norm(uav_v[robotID*2:robotID*2+2]))
	vec.append(uav_v[robotID*2:robotID*2+2])  ## v
	for i in range(totalRigidbodys-1):
		if i == robotID:
			continue
		vec.append(uav_pos[i*2:i*2+2]-uav_pos[robotID*2:robotID*2+2])
	vec.append(np.array(goal)-uav_pos[robotID*2:robotID*2+2])
	print(uav_pos[robotID*2:robotID*2+2])
	rot_mat = matrix._rot_mat_half_pi_minus_theta_counter_clockwise(vec[4])
	for i in range(len(vec)):
		vec[i] = np.dot(rot_mat,vec[i])
	return vec,uav_pos,uav_v,rigidBody


########get the offset angel of lidar ranges with yaw correction#######
def get_angel(dire,yaw):
	theta = (np.arctan2(dire[1],dire[0])+2*np.pi)%(np.pi*2)#ground v
	theta = theta + np.pi/2 - yaw
	return theta%(2*np.pi)    

########get the lidar range######
def get_range(data,dire,yaw):
	offset_angel = get_angel(dire,yaw)
	offset_index = int((offset_angel / 2 / np.pi * len(data))%len(data))
	interval = int(len(data)/30)
	ranges = [data[int(offset_index + i * interval) % len(data)] for i in range(30)]
	for i in range(30):
		if ranges[i] == float('inf'):
			base = offset_index + i * interval
			for j in range(int(interval/2)):
				if data[(base - 1 - j)%len(data)] != float('inf'):
					ranges[i] = data[(base - 1 - j)%len(data)]
					break
				elif data[(base + 1 + j)%len(data)] != float('inf'):
					ranges[i] = data[(base + 1 + j)%len(data)]
					break
			if ranges[i] == float('inf'):
				ranges[i] = 5
		if  ranges[i] > 5:
			ranges[i] = 5
	return ranges

########publish command#######
def pub_command(data):
	global publisher
	v = chassis_control()
	v.x = data[0]
	v.y = data[1]
	publisher.publish(v)

########get observation#######
def get_input(data):
	vec,uav_pos,uav_v,rigidBody = get_vec()
	ranges = get_range(data,vec[0],rigidBody[robotID].euler.z)
	scan_debug = ranges[30 - int(get_angel(vec[0],rigidBody[robotID].euler.z)/2/np.pi*30):30] +\
	           ranges[0 : 30 - int(get_angel(vec[0],rigidBody[robotID].euler.z)/2/np.pi*30)]
	return np.hstack((ranges, vec[0], vec[2], vec[3],vec[4], vec[1])),scan_debug,uav_pos,uav_v,rigidBody

########get center of the robots########
def get_cener(rigidBody):
	return np.array([rigidBody[0].position.x+rigidBody[1].position.x+rigidBody[2].position.x,rigidBody[0].position.y+rigidBody[1].position.y+rigidBody[2].position.y])/3

########get vector using two points#######
def get_rela_goal(rigidBody):
	return np.array([rigidBody[3].position.x-rigidBody[robotID].position.x,rigidBody[3].position.y-rigidBody[robotID].position.y])

########constrain the velovity#########
def velocityclip(v):
	if np.linalg.norm(v) > min_velocity:
		if np.linalg.norm(v) > max_velocity:
			v = v/np.linalg.norm(v)*max_velocity
	else:
		v = np.array([0,0])
	return v

######process lidar#########
def process_lidar(data):
	global state
	input,scan_debug,uav_pos,uav_v,rigidBody = get_input(data.ranges)
	dataout=copy.deepcopy(data)
	dataout.angle_increment = 0.21
	dataout.ranges = scan_debug
	dataout.intensities = [0 for _ in range(30)]
	publisher1.publish(dataout)	
	# Shape adaptation
	obs = np.clip((input - mean) / np.sqrt(var + 1e-8), -10., 10.)
	obs = np.expand_dims(obs, axis=0)
	obs = torch.from_numpy(obs).float().cuda()     # From CPU to GPU
	with torch.no_grad():
		print(is_stop)
		if is_stop:
			state = torch.zeros(1, actor_critic.state_size).cuda()  # Hidden state for RNN
		else:
			# state should be all zeros in the first run
			action, state = actor_critic.act_exe(obs, state, mask)
			cpu_action = action.squeeze(0).cpu().numpy()
			rot_mat = matrix._rot_mat_half_pi_minus_theta_clockwise(get_rela_goal(rigidBody))
			v_out = np.dot(rot_mat,cpu_action)
			v_out = velocityclip(v_out)
			center = get_cener(rigidBody)
			rot_myself = matrix._rot_mat_theta_clockwise(rigidBody[robotID].euler.z)	
			if (center[0]-rigidBody[3].position.x)*(center[0]-rigidBody[3].position.x) + (center[1] - rigidBody[3].position.y)*(center[1]- rigidBody[3].position.y) < goal_error*goal_error:
				print("reach the goal!!!")
				print(rigidBody[0])
				v_out  = np.array([0,0])		
			print(v_out)	
			pub_command(np.dot(rot_myself,v_out))

			# with open('./record/record'+str(rospy.get_time())+'.txt','wb') as f:
			# 	pickle.dump([(goal,uav_pos,uav_v),input[0:30],input,np.array([robotID]),v_out],f)
			# dataout=copy.deepcopy(data)
			# dataout.angle_increment = 0.21
			# dataout.ranges = scan_debug
			# dataout.intensities = [0 for _ in range(30)]
			# publisher1.publish(dataout)

######process opti########
def process_opti(data):
	if data.rigidBodyID > totalRigidbodys - 1:
		pass
	if rigidBody_q[data.rigidBodyID].full():
		rigidBody_q[data.rigidBodyID].get()	
	rigidBody_q[data.rigidBodyID].put(data)

def rc_process(get):
	global is_stop
	if get.assist_mode == 2:
		is_stop = False
	else:
		is_stop = True


if __name__ == "__main__":
	global actor_critic,ob_rms,state,mean,mask
	configuration('/home/nvidia/catkin_ws/src/processor/scripts/robot.ini')
	actor_critic, ob_rms, _ = torch.load(dir_of_model)
	actor_critic.cuda()
	state = torch.zeros(1, actor_critic.state_size).cuda()  # Hidden state for RNN
	mask = torch.ones(1, 1).cuda()     # Termination flag
	mean = ob_rms.mean
	var = ob_rms.var
	obs = np.arange(40)
	obs = np.expand_dims(obs, axis=0)
	obs = torch.from_numpy(obs).float().cuda()     # From CPU to GPU
	actor_critic.act_exe(obs, state, mask)
	print("prepare model done")
	try:                                                                                                               
		rospy.init_node('processor', anonymous=False)
		rospy.Subscriber("scan", LaserScan, process_lidar)
		rospy.Subscriber("agent/opti_odom", Odom, process_opti)
		rospy.Subscriber("/lion_brain/rc", rc, rc_process)
		publisher1 = rospy.Publisher("/scan_debug", LaserScan, queue_size=10)	
		publisher = rospy.Publisher("/lion_brain/chassis_control", chassis_control, queue_size=10)
		rospy.spin()
		print("done")
	except rospy.ROSInterruptException:
		pass
