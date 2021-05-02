# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 15:42:50 2021

@author: Vandan
"""

#pip3 install opencv-python
import cv2
import numpy as np


#  Read Net

net = cv2.dnn.readNet('yolov3_training_last.weights','yolov3_training.cfg')
classes = []

with open('classes.names','r') as f:
    classes = f.read().splitlines() 
    
#img = cv2.imread('images4.jpg')

# for web camera
cap = cv2.VideoCapture(0)

# for video file
#cap = cv2.VideoCapture('demo.mp4')


while True:
    
    _,img = cap.read()
    height, width, _ = img.shape
    
    
    blob = cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0),swapRB=True,crop=False)
    
    # determine only the *output* layer names that we need from YOLO
    output_layers_names = net.getLayerNames()
    output_layers_names = [output_layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    net.setInput(blob)
    layerOutputs = net.forward(output_layers_names)
    
    boxes =[]
    confidences = []
    class_ids = []
    
    for output in layerOutputs:
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
    		# the current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:
                # scale the bounding box coordinates back relative to the
    			# size of the image, keeping in mind that YOLO actually
    			# returns the center (x, y)-coordinates of the bounding
    			# box followed by the boxes' width and height
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                width = int(detection[2]*width)
                height = int(detection[3]*height)
                
                # use the center (x, y)-coordinates to derive the top and
    			# and left corner of the bounding box
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))
                
                
                # update our list of bounding box coordinates, confidences,
    			# and class IDs
                boxes.append([x,y,width,height])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    
    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
    
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255,size=(len(boxes),3))
    
    
    
    
    # ensure at least one detection exists
    if len(indexes)>0:
        # loop over the indexes we are keeping
        for i in indexes.flatten():
            # extract the bounding box coordinates
            x,y,width,height = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img,(x,y),(x+width,y+height),color,2)
            cv2.putText(img,label + " " + confidence, (x,y+20),font,2,(0,0,0),2)
    
    
    cv2.imshow('Image',img)
    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()



#for b in blob:
#    for n, img_blob in enumerate(b):
#        cv2.imshow(str(n),img_blob)




