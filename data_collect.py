#Import Code Dependencies
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math
import mediapipe as mp

#Declare Detection Variables
cap = cv2.VideoCapture(0)
detector = HandDetector(True, maxHands=2)  

#Set Parameter Variables
offset = 20
imgSize = 300

samples = 0
labels = ['paano']

#Camera Feed Loop
for label in labels:
    print(f'Now Capturing {label}')
    while samples < 400:
        success, img = cap.read()
        hands, img = detector.findHands(img, draw=True)
        if hands:
            # Combine bounding boxes of both hands
            x_min = min(hand['bbox'][0] for hand in hands)
            y_min = min(hand['bbox'][1] for hand in hands)
            x_max = max(hand['bbox'][0] + hand['bbox'][2] for hand in hands)
            y_max = max(hand['bbox'][1] + hand['bbox'][3] for hand in hands)
            combined_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

            x, y, w, h = combined_bbox
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            if imgCrop.size > 0:  # Check if imgCrop is not empty
                imgCropShape = imgCrop.shape
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)
            cv2.imshow("Image", img)
        # Saving Logic
        key = cv2.waitKey(1)
        if key == ord('s'):
            samples += 1
            cv2.imwrite(f'Images/{label}/{label}_{samples}.jpg', imgWhite)
            print(samples)
    samples = 0

cap.release()
cv2.destroyAllWindows