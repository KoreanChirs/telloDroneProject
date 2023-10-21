from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from typing import List
import numpy as np
from ultralytics import YOLO
import cv2
from djitellopy import tello
import keypress_module as kp
from time import sleep
import signal
import threading

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

me = tello.Tello()
me.connect()
print(me.get_battery())
kp.init()

move_type = "keyboard"

def custom_function(signal, frame):
    global move_type,ready
    if move_type == "keyboard":
        move_type = "track"
        me.streamon()
    elif move_type == "track":
        move_type = "attack"
    else:
        move_type = "keyboard"
        me.streamoff()
        ready = False
        
signal.signal(signal.SIGINT, custom_function)

pidx = [0.4, 0.4, 0]
pidy = [0.6, 0.6, 0]
pidfb = [0.35, 0.35, 0]  # 0.4 0.4 0
perrorx = 0
perrory = 0
perrorfb = 0
cntt = 0

def track_objects(x,y):
    global perrorx, height, perrory, cntt, perrorfb
    errorx = x - 320 
    errory = 160 - y
    errorfb = 200 - height #target의 크기에 따라 달라야 함. 300
    if cntt == 0:
        perrorx = errorx
        perrory = errory
        perrorfb = errorfb
    speed = pidx[0] * errorx + pidx[1] * (errorx-perrorx)
    yspeed = pidy[0] * errory + pidy[1] * (errory-perrory)
    fbspeed = 40
    #fbspeed = pidfb[0] * errorfb + pidfb[1] * (errorfb - perrorfb)
    if cntt < 30:
        speed = int(np.clip(speed,-30,30))
        yspeed = int(np.clip(yspeed,-50,50))
        fbspeed = int(np.clip(fbspeed,-40,40))
    else:
        speed = int(np.clip(speed,-100,100))
        yspeed = int(np.clip(yspeed,-130,130))
        fbspeed = int(np.clip(fbspeed,-130,130)) 
    fbspeed = 40
    me.send_rc_control(0,fbspeed,yspeed,speed)
    perrorx = errorx
    perrory = errory
    perrorfb = errorfb
    cntt += 1

def attack_objects(x,y):
    global perrorx, height, perrory, cntt
    errorx = x - 320
    errory = 160 - y
    if cntt == 0:
        perrorx = errorx
        perrory = errory
    speed = pidx[0] * errorx + pidx[1] * (errorx-perrorx)
    yspeed = pidy[0] * errory + pidy[1] * (errory-perrory)
    fbspeed = 70
    if cntt < 30:
        speed = int(np.clip(speed,-50,50))
        yspeed = int(np.clip(yspeed,-50,50))
    else:
        speed = int(np.clip(speed,-100,100))
        yspeed = int(np.clip(yspeed,-130,130))
    me.send_rc_control(0,fbspeed,yspeed,speed)
    perrorx = errorx
    perrory = errory
    cntt += 1

def getKeyboardInput():
    lr, fb, ud, yaw = 0,0,0,0
    speed = 30

    if kp.getKey("LEFT"): lr = -speed
    elif kp.getKey("RIGHT"): lr = speed

    if kp.getKey("UP"): fb = speed
    elif kp.getKey("DOWN"): fb = -speed

    if kp.getKey("w"): ud = speed
    elif kp.getKey("s"): ud = -speed

    if kp.getKey("a"): yaw = -speed
    elif kp.getKey("d"): yaw = speed

    if kp.getKey("q"): me.land()
    if kp.getKey("e"):
        try: 
            me.takeoff()
            sleep(3)
        except:
            pass
    return [lr,fb,ud,yaw]


model = YOLO("yolov8x.pt")
model.to('cuda')
CLASS_NAMES_DICT = model.model.names
CLASS_ID = [2,3,5,7]
byte_tracker = BYTETracker(BYTETrackerArgs())
model.predict("test.png")

# settings
LINE_START = Point(50, 1500)
LINE_END = Point(3840-50, 1500)

# Set the resolution of the video

box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)
line_counter = LineCounter(start=LINE_START, end=LINE_END)
line_annotator = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2)

def target():
    global track_id
    track_id = int(input("which person do you want to track?"))

ready = False
track_id = -1
cnt = 0
cnt_detect = 0
target_start = False

while True:
    if move_type == "track" or move_type == "attack":
        frame = me.get_frame_read().frame
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        if ready == False:
            sleep(1)
            ready = True
        frame = cv2.resize(frame,(640,480))
        results = model.predict(frame)
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

        if (2 in detections.class_id) or (3 in detections.class_id) or (5 in detections.class_id) or (7 in detections.class_id):
            cnt_detect = 0
            if track_id == -1:
                me.send_rc_control(0,0,0,0)
                if target_start == False:
                    target_start = True
                    cntt = 0
                    th1 = threading.Thread(target=target)
                    th1.start()
            elif track_id in tracker_id:
                target_start = False
                cnt = 0
                for i in range(len(tracker_id)):
                    if(track_id == tracker_id[i]):
                        idx = i
                try:
                    bottom_right_x = detections.xyxy[idx][0]
                    bottom_right_y = detections.xyxy[idx][1]
                    top_left_x = detections.xyxy[idx][2]
                    top_left_y = detections.xyxy[idx][3]
                    centor_x = (top_left_x + bottom_right_x)/2
                    centor_y = (top_left_y + bottom_right_y)/2
                    height = top_left_y-bottom_right_y
                except:
                    continue
                if move_type == "track":
                    track_objects(centor_x,centor_y)
                else:
                    attack_objects(centor_x,centor_y)
            else:
                target_start = False
                if cnt < 10:
                    cnt += 1
                elif cnt < 20:
                    me.send_rc_control(0,0,0,40)
                    cnt += 1
                else:
                    track_id = -1
        else:
            if cnt_detect < 30:
                cnt_detect += 1
            else:
                me.send_rc_control(0,0,0,40)
                track_id = -1
        cv2.imshow("Video",frame)
        cv2.waitKey(1)
    else:
        vals = getKeyboardInput()
        me.send_rc_control(vals[0],vals[1],vals[2],vals[3])
        sleep(0.05)

