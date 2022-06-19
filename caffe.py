import cv2
import time
import numpy as np
import argparse

# Instance of an 'ArgumentParser' object with a description for this program
parser = argparse.ArgumentParser(description='Caffe SSD object detection')

# Definition of an argument needed to indicate the video to be analyzed
parser.add_argument('-i', '--input_video',
                    help='The video to be analyzed', required=True)

# Converts command line argument values to data types
args = parser.parse_args()

if (args.input_video != 'video_1.mp4') and (args.input_video != 'video_2.mp4') and (args.input_video != 'video_3.mp4'):
    print("ERROR: Didn't include an appropiate input video name.")
    exit()

video_in = 'input/' + args.input_video

if '1' in args.input_video:
    video_out = 'output/video_result_1.mp4'

if '2' in args.input_video:
    video_out = 'output/video_result_2.mp4'

if '3' in args.input_video:
    video_out = 'output/video_result_3.mp4'

# File with a text description of the network architecture
prototxt = 'caffe_files/deploy.prototxt'

# The Caffe SSD model that was pretrained with the COCO dataset
model = 'caffe_files/VGG_coco_SSD_300x300_iter_400000.caffemodel'

# All the classes of objects that the model has been trained to detect
classes = ['person', 'backpack', 'bicycle', 'car', 'traffic light', 'fire hydrant', 'bird', 'cat', 'frisbee', 'skis', 'bottle', 'wine glass', 'banana', 'apple', 'chair', 'couch', 'tv', 'laptop', 'microwave', 'oven', 'book', 'clock', 'umbrella', 'handbag', 'motorcycle', 'airplane', 'stop sign', 'parking meter', 'dog', 'horse', 'snowboard', 'sports ball', 'cup', 'fork', 'sandwich', 'orange', 'potted plant', 'bed', 'mouse', 'remote',
           'toaster', 'sink', 'vase', 'scissors', 'tie', 'suitcase', 'bus', 'train', 'bench', 'sheep', 'cow', 'kite', 'baseball bat', 'knife', 'spoon', 'broccoli', 'carrot', 'dining table', 'toilet', 'keyboard', 'cell phone', 'refrigerator', 'teddy bear', 'hair drier', 'truck', 'boat', 'elephant', 'bear', 'baseball glove', 'skateboard', 'bowl', 'hot dog', 'pizza', 'toothbrush', 'zebra', 'giraffe', 'surfboard', 'tennis racket', 'donut', 'cake']

# The 'uniform' function will return a Numpy ndarray the same size as 'classes'
# that will hold a color for each object class (The parameters mean: choose a
# number between 0 and 255 three times and save the 3 numbers as an element in
# the ndarray)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Reads a network model stored in the Caffe framework's format
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Create a VideoCapture object to read from the given video file
cap = cv2.VideoCapture(video_in)

# Get the video frames' width and height for proper saving of videos
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Create a VideoWriter object to write an output video with the given specs
out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(
    *'mp4v'), 30, (frame_width, frame_height))

# Detect objects in each frame of the video
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        image = frame

        # HERE
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

        H, S, V = cv2.split(hsv_image)

        eq_H = cv2.equalizeHist(H)
        eq_S = cv2.equalizeHist(S)
        eq_V = cv2.equalizeHist(V)

        hsv_image_eq = cv2.merge([H, eq_S, eq_V])
        image_eq_rgb = cv2.cvtColor(hsv_image_eq, cv2.COLOR_HSV2RGB)

        h, w = image_eq_rgb.shape[:2]

        # mean: A scalar with mean values which are subtracted from each color channel
        # scalefactor: A float that represents how zoomed in the image is
        blob = cv2.dnn.blobFromImage(image=rgb_image, size=(
            300, 300), scalefactor=0.07, swapRB=False, mean=(50, 50, 50))

        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

                (startX, startY, endX, endY) = box.astype("int")

                label = "{}: {:.2f}%".format(classes[idx], confidence * 100)

                cv2.rectangle(image_eq_rgb, (startX, startY),
                              (endX, endY), colors[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image_eq_rgb, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

        # If the frame is converted to rgb, change it to bgr here
        bgr_image = cv2.cvtColor(image_eq_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow('image', bgr_image)

        out.write(bgr_image)  # Write the frame into the output video file

        # Press 'q' to exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
out.release()

cv2.destroyAllWindows()
