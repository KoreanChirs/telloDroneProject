from djitellopy import tello
import keypress_module as kp
from time import sleep
import numpy as np
import cv2
import math 
import time
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from typing import List
from ultralytics import YOLO
import keypress_module as kp
import bluetooth
import signal

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


model = YOLO("yolov8n.pt")
CLASS_NAMES_DICT = model.model.names
CLASS_ID = [0]

# settings
LINE_START = Point(50, 1500)
LINE_END = Point(3840-50, 1500)

# Set the resolution of the video

box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)
line_counter = LineCounter(start=LINE_START, end=LINE_END)
line_annotator = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2)


f_x = 450
f_y = 450

forward_speed = 30 # cm/s (real speed) 
angular_speed = 36 # Degrees/s  

me = tello.Tello()
me.connect()
print(me.get_battery())
kp.init()
me.streamon()
x,y = -50,0
real_y = -y
move_angle = 0 # direction of velocity (not forward angle)
ya = 0 #forward angle

# def handler(signum, frame):
#     try:
#         me.land(); sleep(3)
#     except:
#         pass

# signal.signal(signal.SIGINT, handler)

def getKeyboardInput():
    global ya, x, y, move_angle, Interval
    lr, fb, ud, yaw = 0,0,0,0
    d = 0

    dInterval = forward_speed * Interval

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
    elif kp.getKey("d"): 
        yaw = forward_speed

    if kp.getKey("q"): me.land(); sleep(3)

    try: 
        if kp.getKey("e"): me.takeoff()
    except:
        pass

    move_angle += ya
    x += int(d*math.cos(math.radians(move_angle)))
    y += int(d*math.sin(math.radians(move_angle)))
    return [lr,fb,ud,yaw]

ya_offset = me.get_yaw()

img = np.zeros((1000,1000,3), np.uint8)

def drawPoints():
    cv2.circle(img,(x+500,y+500), 5 ,(0,0,255),cv2.FILLED) #B,G,R

server_address = "F4:46:37:AE:61:F8"
port = 29

sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
sock.connect((server_address,port))

last = time.time()

while True:
    frame = me.get_frame_read().frame
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame,(640,480))
    results = model.predict(frame, show = True)
    frame = cv2.resize(frame,(640,480))
    detections = Detections(
        xyxy=results[0].boxes.xyxy.cpu().numpy(),
        confidence=results[0].boxes.conf.cpu().numpy(),
        class_id=results[0].boxes.cls.cpu().numpy().astype(int)
    )
    real_y = -y
    now = time.time()
    Interval = now - last
    last = now
    ya = me.get_yaw() - ya_offset
    vals = getKeyboardInput()
    me.send_rc_control(vals[0],vals[1],vals[2],vals[3])
    if 0 in detections.class_id:
        bottom_right_x = detections.xyxy[0][0]
        bottom_right_y = detections.xyxy[0][1]
        top_left_x = detections.xyxy[0][2]
        top_left_y = detections.xyxy[0][3]
        left_x = (top_left_x + bottom_right_x)/2
        left_y = (top_left_y + bottom_right_y)/2

        left_beta = math.atan((left_x - 320)/f_x)
        left_beta = math.degrees(left_beta)

        left_vertical_alpha = math.atan((240 - left_y)/f_y)
        left_vertical_alpha = math.degrees(left_vertical_alpha)

        theta = -(left_beta + ya) + 90 #x, real_y, theta
        try:
            sock.send("{} {} {} {}".format(time.time(), x,real_y,theta))
        except: 
            me.land(); sleep(3)
    #print(x,real_y)
    #cv2.imshow("Video",frame)
   # drawPoints()
    #cv2.imshow("img", img)
    #cv2.waitKey(1)