from djitellopy import tello
import keypress_module as kp
from time import sleep
import numpy as np
import cv2
import math 
# parameters

# 화면 내의 speed와 실제 드론의 speed를 같게 맞출 이유는 없음. 
# 드론 날리는 공간 크기 잘 고려해서 화면 속도랑 드론 속도 맞추기. 

forward_speed = 30 # cm/s (real speed) 
angular_speed = 36 # Degrees/s  
interval = 0.25 # s

dInterval = forward_speed * interval
aInterval = angular_speed * interval

me = tello.Tello()
me.connect()
print(me.get_battery())
kp.init()

x,y = 500,500
move_angle = 0 # direction of velocity (not forward angle)
ya = 0 #forward angle

#좌표계 좌표가 내가 생각하는 좌표랑 달라서 각도가 좀 이상함. 
def getKeyboardInput():
    global ya, x, y, move_angle
    lr, fb, ud, yaw = 0,0,0,0
    d = 0

    if kp.getKey("LEFT"): 
        lr = -forward_speed
        d = dInterval
        move_angle = -180

    elif kp.getKey("RIGHT"): 
        lr = forward_speed
        d = -dInterval
        move_angle = 180

    if kp.getKey("UP"): 
        fb = forward_speed
        d = dInterval
        move_angle = 270

    elif kp.getKey("DOWN"): 
        fb = -forward_speed
        d = -dInterval
        move_angle = -90
     
    if kp.getKey("w"): ud = forward_speed
    elif kp.getKey("s"): ud = -forward_speed

    if kp.getKey("a"): 
        yaw = -forward_speed
        ya -= aInterval
    elif kp.getKey("d"): 
        yaw = forward_speed
        ya += aInterval

    if kp.getKey("q"): me.land(); sleep(3)
    if kp.getKey("e"): me.takeoff()

    sleep(interval)
    move_angle += ya
    x += int(d*math.cos(math.radians(move_angle)))
    y += int(d*math.sin(math.radians(move_angle)))

    return [lr,fb,ud,yaw]

def drawPoints():
    cv2.circle(img,(x,y), 5 ,(0,0,255),cv2.FILLED) #B,G,R

img = np.zeros((1000,1000,3), np.uint8)


while True:
    vals = getKeyboardInput()
    me.send_rc_control(vals[0],vals[1],vals[2],vals[3])
    drawPoints()
    cv2.imshow("Output",img)
    cv2.waitKey(1)
    

