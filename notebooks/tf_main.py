import numpy as np
import tensorflow as tf
import cv2 as cv2
from tensorflow.keras import models, layers 
from tensorflow.keras.utils import to_categorical 
import argparse


parser = argparse.ArgumentParser(description='Caffe SSD object detection')

parser.add_argument('-i', '--input_video', type=str, default='input/video_1.mp4', help='input video file')

args = parser.parse_args()

if (args.input_video != 'video_1.mp4') and (args.input_video != 'video_2.mp4') and (args.input_video != 'video_3.mp4'):
    print("ERROR: Didn't include an appropiate input video name.")
    exit()

video_in = 'input/' + args.input_video

with open('models/object_detection_classes_coco.txt', 'r') as f:
    class_names = f.read().split('\n')

# get a different color array for each of the classes
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

# load the DNN model
mobile_net_model = cv2.dnn.readNet(model='models/frozen_inference_graph.pb',
                        config='models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', 
                        framework='TensorFlow')


def preprocessImg(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (700, 700))
    image = equalize_RGB(image)
    return image

def drawDetections(output, img, copy_image):
    image_height, image_width, _ = copy_image.shape
    # loop over every detection
    for detection in output[0, 0, :, :]:
        confidence = detection[2]
        if confidence > .3:
            class_id = detection[1]
            class_name = class_names[int(class_id)-1]
            color = COLORS[int(class_id)]
            box_x = detection[3] * image_width
            box_y = detection[4] * image_height

            box_width = detection[5] * image_width
            box_height = detection[6] * image_height

            cv2.rectangle(copy_image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)

            cv2.putText(copy_image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(copy_image, str(round(confidence, 2)), (int(box_x + 120), int(box_y -5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return copy_image

def getBlob(img):
    # get the blob from the image
    red_values_mean = np.mean(img[:,:,0])
    green_values_mean = np.mean(img[:,:,1])
    blue_values_mean = np.mean(img[:,:,2])
    blob = cv2.dnn.blobFromImage(image=img, size=(700,700), mean=(red_values_mean, green_values_mean, blue_values_mean))
    return blob

def getFrameWithBoxes(img):
    copy_image = img.copy()
    processed_img = preprocessImg(img)

    blob = getBlob(processed_img)
    
    mobile_net_model.setInput(blob)

    output = mobile_net_model.forward()

    drawDetections(output, processed_img, copy_image)
    
    return copy_image

def equalize_RGB(img):
    # convert to YUV color space
    red = img[:,:,0]
    green = img[:,:,1]
    blue = img[:,:,2]
    # equalize the red, green, and blue channels
    red_eq = cv2.equalizeHist(red)
    green_eq = cv2.equalizeHist(green)
    blue_eq = cv2.equalizeHist(blue)
    # merge the channels back together
    img_eq = cv2.merge((red_eq, green_eq, blue_eq))
    return img_eq

cap = cv2.VideoCapture(video_in)

# get the video frames' width and height for proper saving of videos
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# create the `VideoWriter()` object
out = cv2.VideoWriter('output/video_result_1.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))


# detect objects in each frame of the video
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        image_eq = equalize_RGB(frame)
        image_boxes = getFrameWithBoxes(image_eq)

        cv2.imshow('image', image_boxes)
        out.write(image_boxes)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()