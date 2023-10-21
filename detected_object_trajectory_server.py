from djitellopy import tello
import keypress_module as kp
from time import sleep
import numpy as np
import cv2
import math 
import time
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from typing import List
from ultralytics import YOLO
import keypress_module as kp
import threading
from collections import deque
import bluetooth

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

# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)

# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections, 
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)
    
    tracker_ids = [None] * len(detections)
    
    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids


model = YOLO("yolov8x.pt")
model.to('cuda')
CLASS_NAMES_DICT = model.model.names
CLASS_ID = [0]
byte_tracker = BYTETracker(BYTETrackerArgs())
model.predict("test.png")

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
angular_speed = 60 # Degrees/s  

me = tello.Tello()
me.connect()
print(me.get_battery())
kp.init()
me.streamon()

x,y = 50,0
real_y = -y
move_angle = 0 # direction of velocity (not forward angle)
ya = 0 #forward angle

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
    if kp.getKey("e"): me.takeoff()

    move_angle += ya
    x += int(d*math.cos(math.radians(move_angle)))
    y += int(d*math.sin(math.radians(move_angle)))
    return [lr,fb,ud,yaw]

def drawPoints():
    global x_cross,y_cross
    cv2.circle(img,(x_cross+500,-y_cross+800), 5 ,(0,0,255),cv2.FILLED)

img = np.zeros((1000,1000,3), np.uint8)

server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)

port = 29
server_sock.bind(("", port))
server_sock.listen(1)

file = open('result.txt','w')

print("Listening for connections on RFCOMM channel {}".format(port))

client_sock, client_info = server_sock.accept()
print("Accepted connection from ", client_info)

ya_offset = me.get_yaw()

x_cross = 0
y_cross = 0

def th1():
    global x,real_y,theta,height_tan,img,file,x_cross,y_cross
    while True:
        try:
            data = client_sock.recv(1024)
            data = data.decode('utf-8')
            data = str(data).split(" ")
            data = list(map(float,data)) # error
            t_m, x_m,y_m,theta_m, height_tan = [int(time.time()), x,real_y,theta,height_tan]
            t_o,x_o,y_o,theta_o = data
            theta_m = math.tan(math.radians(theta_m))
            theta_o = math.tan(math.radians(theta_o))
            x_cross = int((theta_m*x_m-theta_o*x_o-y_m+y_o)/(theta_m-theta_o))
            y_cross = int(theta_m*(x_cross-x_m) + y_m)
            a = math.dist([(x_m + x_o)/2,(y_m + y_o)/2],[x_cross,y_cross])
            print(x_cross,y_cross,a*height_tan + me.get_height())
            file.write(f"{x_cross} {y_cross} {a*height_tan + me.get_height()}\n")
            file.flush()
        except:
            pass  

thread1 = threading.Thread(target=th1)

thread1.start()

last = time.time()

cnt = 0
while True:
    frame = me.get_frame_read().frame
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame,(640,480))
    results = model.predict(frame)
    frame = cv2.resize(frame,(640,480))
    detections = Detections(
        xyxy=results[0].boxes.xyxy.cpu().numpy(),
        confidence=results[0].boxes.conf.cpu().numpy(),
        class_id=results[0].boxes.cls.cpu().numpy().astype(int)
    )
    # filtering out detections with unwanted classes
    mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
    detections.filter(mask=mask, inplace=True)
    tracks = byte_tracker.update(
        output_results=detections2boxes(detections=detections),
        img_info=frame.shape,
        img_size=frame.shape
    )
    tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
    detections.tracker_id = np.array(tracker_id)
    mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
    detections.filter(mask=mask, inplace=True)
    # format custom labels
    labels = [
        f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id
        in detections
    ]
    # updating line counter
    line_counter.update(detections=detections)
    frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
    line_annotator.annotate(frame=frame, line_counter=line_counter)
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
        height_tan = math.tan(math.radians(left_vertical_alpha))
    drawPoints()
    cv2.imshow("img",img)
    cv2.waitKey(1)
    #print(x,real_y)
    #cv2.imshow("Video",frame)
    #cv2.waitKey(1)
    #cv2.imshow("Output",img)
    #cv2.waitKey(1)
    #print(x,real_y)
    #drawPoints()
    #print(me.get_height())
file.close()
