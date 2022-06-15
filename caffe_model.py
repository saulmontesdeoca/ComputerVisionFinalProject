import cv2
import time
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Caffe model file')

parser.add_argument('-i', '--input_video',
                    help='The video to be analyzed', required=True)

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
