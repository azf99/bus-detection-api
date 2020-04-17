from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import cv2
import time
import os
import sys
import json
from numpy import expand_dims, argmax, asarray

class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        #start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        #end_time = time.time()

      #  print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()


    
model_path = 'ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb'
odapi = DetectorAPI(path_to_ckpt=model_path)
    
####Threshold for person detection
threshold = 0.2


app = Flask(__name__)

CORS(app)

@app.route('/me', methods = ["GET", "POST"])
def hello_world():
    return jsonify('Hello, World!')



@app.route('/status', methods = ["POST"])
def predict():
    
    path = request.form["base"]
    #print(path)
    
    files = os.listdir(path)
    res = []
    area = []
    #print(path)
    files.sort()
    #print(len(files))
    
    #filestest = [files[0], files[1]]
    st = time.time()
    
    #print(files)
    for file in files:
        
        come = 0
        go = 0

        #frame = cv2.resize(frame, (1280, 720))
        
        frame = cv2.imread(os.path.join(path, file))
        frame = cv2.resize(frame, (400, 250))
        
        boxes, scores, classes, num = odapi.processFrame(frame)
        #print("classes: ", classes[:3])
        #print("num = ", num)
        # Visualization of the results of a detection.
        for i in range(0, num):
            
            # Class 1 represents human
            if classes[i] == 3 or classes[i] == 44 or classes == 6 or classes == 8:
                #print(file)
                a = (boxes[i][0]+boxes[i][2])*(boxes[i][1]*boxes[i][3])
                if a > (0.08 * 400 * 250):
                    area.append(area)
                    res.append(boxes[i][2]) 
                    
                else:
                    continue
            
    for i in range(1, len(res)):
 
            if res[i-1] < res[i]:
                come+=1
            else:
                go+=1
    #print("come, go=", come, go)
    
    '''
    for i in range(1, len(area)):
 
            if area[i-1] < area[i]:
                come+=1
            else:
                go+=1
    
    text = ""
    print("come, go=", come, go)
    '''
    if come>go:
        text = "arrival"
        #print("Come")
    else:
        text = "departure"
        #print("Go")
    #f = np.array(res[0]) - np.array(res[1])
    #print(res)
    #print(f)
    print("time taken:", time.time() - st)
    
    text = json.dumps({"Status": text})
    
    return text

app.run(threaded = True, port = int(sys.argv[1]))
