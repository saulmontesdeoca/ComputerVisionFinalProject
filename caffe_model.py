import cv2
import time
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Caffe model file')

parser.add_argument('-v', '--analyze_video',
                    help='The video to be analyzed', required=True)

args = parser.parse_args()

if args.analyze_video != 'input/video_1.mp4' and args.analyze_video != 'input/video_2.mp4' and args.analyze_video != 'input/video_3.mp4':
    print("ERROR: Didn't include a video name.")
    exit()

# Create a VideoCapture object to read from the given video file
cap = cv2.VideoCapture('input/video_1.mp4')

# Get the video frames' width and height for proper saving of videos
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Create a VideoWriter object to write an output video with the given specs
out = cv2.VideoWriter('output/video_result_1.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

# Detect objects in each frame of the video
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        image = frame

        # HERE

        cv2.imshow('image', image)
        out.write(image)  # Write the frame into the output video file

        # Press 'q' to exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
out.release()

cv2.destroyAllWindows()
