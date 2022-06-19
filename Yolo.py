import cv2
import time
import argparse
import numpy as np

# Code to generate the model
with open("files/object_detection_classes_coco.txt", "r") as f:
    class_names = f.read().split("\n")

# get a different color array for each of the classes
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

# load the DNN model
yolo_model = cv2.dnn.readNetFromDarknet("files/yolov3.cfg", "files/yolov3.weights")

ln = yolo_model.getLayerNames()
ln = [ln[i - 1] for i in yolo_model.getUnconnectedOutLayers()]

print(ln)


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="Archivo de video")
args = vars(ap.parse_args())


cap = cv2.VideoCapture(args["video"])

# get the video frames' width and height for proper saving of videos
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# create the `VideoWriter()` object
out = cv2.VideoWriter(
    "output/video_result_1.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (frame_width, frame_height),
)

# detect objects in each frame of the video
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        image = frame

        # equalize the image
        colorimage_r = cv2.equalizeHist(image[:, :, 2])
        colorimage_g = cv2.equalizeHist(image[:, :, 1])
        colorimage_b = cv2.equalizeHist(image[:, :, 0])

        # merge the channels
        colorimage_e = cv2.merge((colorimage_b, colorimage_g, colorimage_r))

        (
            image_height,
            image_width,
            _,
        ) = colorimage_e.shape  # get height and width of image

        # add blob to the preprocessed image
        blob = cv2.dnn.blobFromImage(colorimage_e, 1 / 255.0, (416, 416), crop=False)
        yolo_model.setInput(blob)
        layerOutputs = yolo_model.forward(ln)

        # define the variables for the bounding boxes, class names, and confidence scores
        confidences = []
        classIDs = []
        boxes = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > 0.7:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array(
                        [image_width, image_height, image_width, image_height]
                    )
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

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(colorimage_e, (x, y), (x + w, y + h), color, 2)
                text = f"{class_names[classIDs[i]]}"
                cv2.putText(
                    colorimage_e,
                    text,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        cv2.imshow("image", colorimage_e)
        out.write(colorimage_e)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
