MASK DETECTION USING YOLOV3 ARCHITECTURE

1) Data Collection
  -> Webscraped the images of people with and without mask from google.
  -> file: images are available in images folder.

2) Data Preparation
  -> Created annotations on images to get coordinates of Object within image using labelimg tool.
  -> files: txt files containing annotation are in images folder.
  
3) Model Training
  -> Trained a model using yolov3 architecture on google colab.
  -> files: yolo_training.ipynb
  
4) Detection
  -> Detects person with mask and without mask accurately.
  -> file: Yolo.py


