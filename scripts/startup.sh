#! /bin/bash 
source /opt/ros/kinetic/setup.sh 
source /home/nvidia/catkin_ws/devel/setup.sh
gnome-terminal  "roscore" -x bash -c "roscore;exec bash;"
sleep 5s
gnome-terminal  "process_lidar" -x bash -c "roslaunch rplidar_ros view_rplidar.launch;exec bash;"
sleep 5s
gnome-terminal  "process_opti" -x bash -c "rosrun bebop_odom opti_odom.py;exec bash;"
sleep 5s
gnome-terminal  "lion_brain" -x bash -c "roslaunch lion_brain lion_brain.launch;exec bash;"
sleep 5s
rosrun processor process_lidar.py --load-dir /home/nvidia/catkin_ws/src/processor/scripts/models/model1.pt



