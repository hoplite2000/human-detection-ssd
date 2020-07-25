from utils.centroidtracker import CentroidTracker
from utils.trackableobject import TrackableObject
from utils.keyclipwriter import KeyClipWriter
from imutils.video import FPS
import numpy as np
import imutils
import time
import dlib
import cv2
import sys
import os
import datetime

prototxt = './mobilenet_ssd/MobileNetSSD_deploy.prototxt'
model = './mobilenet_ssd/MobileNetSSD_deploy.caffemodel'
conf = 0.3
inputfilepath = ''
outputfilepath = ''
frame_width = 700
linex1, linex2 = 0, frame_width
liney1, liney2 = None, None
consecFrames = 0
record = False
c = 0
bufSize = 128

if len(sys.argv) > 1:
    inputfilepath = sys.argv[1]
    outputfilepath = sys.argv[2]

#create output folder if it does not exists
if not os.path.exists(outputfilepath):
    os.mkdir(outputfilepath)

# initialize key clip writer and
kcw = KeyClipWriter(bufSize)

# initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("> Loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# initialize the video stream and pointer to output video file
print("> Accessing video stream...")
vs = cv2.VideoCapture(inputfilepath if inputfilepath != "" else 0)
#vs = cv2.VideoCapture('pedestrians.mp4')
time.sleep(1)

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

totalFrames = 0
totalDown = 0
totalUp = 0
totalUptemp = 0
totalDowntemp = 0

# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while True:

    (ret, frame) = vs.read()

    if not ret:
        break

    frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if W is None or H is None:
        (H, W) = frame.shape[:2]
        liney1, liney2 = H // 2, H // 2

    rects = []

    if totalFrames % 3 == 0:
        trackers = []

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > conf:

                idx = int(detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                trackers.append(tracker)

                cX = (endX+startX)//2
                cY = (endY+startY)//2

                eqn = ((liney2 - liney1) / (linex2 - linex1)) * (endX - linex1) + liney1
                if endY <= eqn:
                    color = (0, 255, 0)
                    colorc = (0, 0, 255)
                else:
                    color = (0, 0, 255)
                    colorc = (0, 255, 0)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.circle(frame, (cX, cY), 5, colorc, -1)

    else:
        for tracker in trackers:

            tracker.update(rgb)
            pos = tracker.get_position()

            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            rects.append((startX, startY, endX, endY))

            cX = (endX + startX) // 2
            cY = (endY + startY) // 2

            eqn = ((liney2 - liney1) / (linex2 - linex1)) * (endX - linex1) + liney1
            if endY <= eqn:
                color = (0, 255, 0)
                colorc = (0, 0, 255)
            else:
                color = (0, 0, 255)
                colorc = (0, 255, 0)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, colorc, -1)

    cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

    objects = ct.update(rects)

    for (objectID, centroid) in objects.items():

        to = trackableObjects.get(objectID, None)

        if to is None:
            to = TrackableObject(objectID, centroid)

        else:

            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            if not to.counted:

                if direction < 0 and (H//2)-10 < centroid[1] < H // 2:
                    totalUp += 1
                    totalUptemp+=1
                    to.counted = True

                elif direction > 0 and (H//2)+10 > centroid[1] > H // 2:
                    totalDown += 1
                    totalDowntemp+=1
                    to.counted = True

        trackableObjects[objectID] = to

    # recordings
    if record:
        if c > 10:
            record = False
            c = 0
            totalUptemp = 0
            totalDowntemp = 0
        # if any problems occurs like capturing video even if no intruders are there then change the below condition to 2
        if totalDowntemp > 1 or totalUptemp > 1:
            if c < 10:
                timestamp = datetime.datetime.now()
                p = "{}/{}.avi".format(outputfilepath, timestamp.strftime("%Y%m%d-%H%M%S"))
                kcw.start(p, cv2.VideoWriter_fourcc(*"MJPG"), 25)
                c += 1
            else:
                kcw.finish()
                totalUptemp = 0
                totalDowntemp = 0
                c = 0
                record = False
    else:
        totalUptemp = 0
        totalDowntemp = 0
        c = 0

    # construct a tuple of information we will be displaying on the
    # frame
    info = [
        ("Up(intruders)", totalUptemp),
        ("Down(intruders)", totalDowntemp),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    consecFrames += 1

    kcw.update(frame)

    if kcw.recording and consecFrames == bufSize:
        kcw.finish()
        totalUptemp = 0
        totalDowntemp = 0

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `esc` key was pressed, break from the loop
    if key == 27:
        break

    totalFrames += 1

    sys.stdout.flush()
    sys.stdout.write("\r> Above line: {} and Below line: {} ".format(totalUp, totalDown))
    sys.stdout.flush()

    # increment the total number of frames processed thus far and
    # then update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("\n> Elapsed time: {:.2f}".format(fps.elapsed()))
print("> Approx. FPS: {:.2f}".format(fps.fps()))

vs.release()
cv2.destroyAllWindows()