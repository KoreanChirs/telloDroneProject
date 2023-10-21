# telloDroneProject
Creating a drone which follows certain object we want. (also get coordinate of that object in realtime)

1. using YOLO and BYTETRACK algorithm, i make a drone which follows certain object we want. [reconnaissance_attack.py]
   - system have 4 state.
   - first state is user control mode where we can control the drone with keyboard.
   - second state is reconnaissance state where drone is spinning until discovering the instance that belongs to our reserved class. (and we can set up the certain instance as target with tracking id)
   - third state is tracking state where drone is tracking the target. (using pid control)
   - fourth state is attacking mode where drone makes collision with target.

2. using YOLO and BYTETRACK algorithm, i make a drone system that can get coordinate of target. [detected_object_trajectory_client, detected_object_trajectory_server]
   - using two drones, we can get 3D coordinate of target.
   - each drone communicates with a laptop using wifi socket. (one to one mapping from drone to laptop)
   - between laptops, they are communicating with bluetooth socket.
   - Currently, the keyboard control mode only supports it, but with a small code modification, it will be possible in all modes. (because algorithm is quite simple)

3. version information
   - ultralytics(YOLO): 8.0.10
   - numpy : 1.22.4
   - torch : torch-1.13.1+cu116

4. the websites that can help me address version issues
   - https://stackoverflow.com/questions/60349980/is-there-a-way-to-install-cython-bbox-for-windows
   - https://pytorch.org/get-started/previous-versions/

5. videos and websites which help me a lot.
   - https://www.youtube.com/watch?v=gRAyOPjQ9_s   {object detection}
   - https://universe.roboflow.com/search?q=tank   {dataset}
   - https://github.com/ultralytics/ultralytics    {yolo algorithm}
   - https://www.computervision.zone/courses/drone-programming-course/  {tello drone api reference}
     
