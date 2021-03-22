#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
from timeit import time
import warnings
import os
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings('ignore')
#from keras import backend as K
#K.tensorflow_backend._get_available_gpus()


def main(yolo):
    #start = time.time()
   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
    H = 0
    W = 0
    
   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    #input_video ='rtsp://admin:123@192.168.0.188:554'
    input_video ="0"

    
    writeVideo_flag = True 

    video_capture = cv2.VideoCapture(input_video)
    #count person id
    counter = []
    
    writeVideo_flag = True
    if writeVideo_flag:
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter("test.avi", fourcc, 30, (w, h))
    
    
    
    fps = 0.0
    while True:
        
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        #frame = video_capture.read()
        t1 = time.time()
        (H, W) = frame.shape[:2]
        image = Image.fromarray(frame) #bgr to rgb
        boxs = yolo.detect_image(image)
        # print("box_num",len(boxs))
        features = encoder(frame,boxs)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        tracker.predict()
        tracker.update(detections)
        
        i = int(0)
        #Detection Person
        for det in detections:
            bbox = det.to_tlbr()
            #cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
        #Tracking Preson
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            counter.append(int(track.track_id))
            i += 1
            """
            center_x = int(((bbox[0])+(bbox[2]))/2)
            center_y = int(((bbox[1])+(bbox[3]))/2)
            
            cv2.circle(frame, (center_x,center_y), 4, (0, 255, 0), -1)
            """
            
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(0,255,0), 2)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
            
            
        count = len(set(counter))
        #cv2.putText(frame, "Total People Counter: "+str(count),(int(20), int(120)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        #cv2.putText(frame, "Current People Counter: "+str(i),(int(20), int(80)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        #cv2.putText(frame, "People Counts: "+str(count),(int(20), int(60)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        #cv2.putText(frame, "FPS: %f"%(fps),(int(20), int(40)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        
        cv2.imshow('test', frame)
        
        if writeVideo_flag:
            # save a frame
            out.write(frame)
        
        
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        #print("fps= %f"%(fps))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
     
    #video_capture.stop()
    if writeVideo_flag:
        out.release()
     
    #end = time.time()
    #spend = str(int(end - start))
    #print("Process Processing completed: ", FILENAME)
    #outfile = open("./output/" + FILENAME +".txt", "w")
    #outfile.write("Total People Counter: " + str(count))
    #outfile.write("\nSpend time: " + spend + " seconds")
    #outfile.close()
    #print("Save number is completed")
    video_capture.release()
    cv2.destroyAllWindows()
    
    #print("Spend time: " + spend + " seconds")

if __name__ == '__main__':
    
    main(YOLO(gpu_memory=1))
    