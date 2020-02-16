# People Count

# import the necessary packages
import numpy as np
import random
from datetime import datetime
import time
import json
import cv2


inpWidth = 288            # Width of network's input image
inpHeight = 288           # Height of network's input image
objectnessThreshold = 0.5 # Objectness threshold
confThreshold = 0.5       # Confidence threshold
nmsThreshold = 0.4        # Non-maximum suppression threshold
count = 0
l_people = [False, False, False]
dict_file = {}



# Get the names of the output layers

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Load names of classes
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')



# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
print(classes)



# Read video
video = cv2.VideoCapture("stock_footage.webm")

out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (596, 336))

# Exit if video not opened
if not video.isOpened():
    print("Could not open video")

# Read first frame
ok, frame = video.read()
if not ok:
    print('Cannot read video file')

while True:
    
    # Read a new frame
    start_time = time.time()
    ok, frame = video.read()
    if not ok:
        break
    
    
    # Load our input image and grab its spatial dimensions
    (H, W) = frame.shape[:2]
    print(H, W)
    
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    # Sets the input to the network
    net.setInput(blob)
      
    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))
    
    boxes = []
    confidences = []
    classIDs = []
    text = "Count : {}".format(count)
    cv2.putText(frame, text, (500,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    for output in outs:
        # loop over each of the detections
    
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            #print(classes[classID])
           
            if classes[classID] != "person":
                continue
            #print('enter')
            
            if confidence > objectnessThreshold:
    			# scale the bounding box coordinates back relative to the
    			# size of the image, keeping in mind that YOLO actually
    			# returns the center (x, y)-coordinates of the bounding
    			# box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                print(box)
                (centerX, centerY, width, height) = box.astype("int")
                
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
     
    			# update our list of bounding box coordinates, confidences,
    			# and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    print(idxs)
    
    # ensure at least one detection exists
    if len(idxs) > 0:
        count = len(idxs)
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            #centerCoord = (x+(w/2), y+(h/2))
            #print(centerCoord)
            
            # draw a bounding box rectangle and label on the image
            color = random.randint(0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "ID : {}".format(i+1)
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            
    #show the output image
    cv2.imshow('frame', frame)
    out.write(frame)
    # Break if ESC pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break
    print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop
    date = str(datetime.now())
    print(date)
    dict_file[date] = 'Count: ' + str(count)

with open('output.json', 'w') as fp:
    json.dump(dict_file, fp)

out.release()
video.release()
cv2.destroyAllWindows()


 


