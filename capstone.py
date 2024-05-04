# Import necessary libraries
import cv2
import numpy as np
import math
import playsound
import os
from cvzone.HandTrackingModule import HandDetector
import tflite_runtime.interpreter as tflite
from threading import Thread

# Declare Detection and Object Variables
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.45)  # Adjust detection confidence
interpreter = tflite.Interpreter(model_path=os.getcwd() + "/Model/keras_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set Parameters and Storage Variables
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
offset = 20
img_size = 224
frame_int = 20
labels = [ 'ako',
           'ano',
           'aral',
           'bahay',
           'bakit',
           'barya',
           'gusto',
           'gutom',
           'ikaw',
           'kailan',
           'laro',
           'nanay',
           'oras',
           'paano',
           'pagkain',
           'pamilya',
           'patawad',
           'pera',
           'saan',
           'sakit',
           'salamat',
           'sila',
           'tatay',
           'tubig',
           'tulong',
          ]

sound_files = {
    'ako': 'ako.mp3',
    'ano': 'ano.mp3',
    'aral':'aral.mp3',
    'bahay': 'bahay.mp3',
    'bakit': 'bakit.mp3',
    'barya': 'barya.mp3',
    'gusto': 'gusto.mp3',
    'gutom': 'gutom.mp3',
    'ikaw': 'ikaw.mp3',
    'kailan': 'kailan.mp3',
    'laro': 'laro.mp3',
    'nanay': 'nanay.mp3',
    'oras': 'oras.mp3',
    'paano': 'paano.mp3',
    'pagkain': 'pagkain.mp3',
    'pamilya': 'pamilya.mp3',
    'patawad': 'patawad.mp3',
    'pera': 'pera.mp3',
    'saan': 'saan.mp3',
    'sakit': 'sakit.mp3',
    'salamat': 'salamat.mp3',
    'sila': 'sila.mp3',
    'tatay': 'tatay.mp3',
    'tubig': 'tubig.mp3',
    'tulong': 'tulong.mp3',

}

words = []
current_word = ""

# Define a function for hand detection and classification
def process_frame(frame):
    global words, current_word

    hands, img= detector.findHands(frame)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        img_white = np.ones((img_size, img_size, 3), np.uint8)*255
        img_crop = frame[y - offset:y + h + offset, x - offset :x + w + offset]
        aspect_ratio = h / w

        if aspect_ratio > 1:
            try:
                k = img_size/h
                width_calc = math.ceil(k * w)
                img_resize = cv2.resize(img_crop, (width_calc, img_size))
                width_gap = math.ceil((img_size - width_calc)/2)
                img_white[:, width_gap:width_gap + width_calc] = img_resize
                #img_white = cv2.Canny(img_white, 50, 100)
                #img_white = cv2.cvtColor(img_white, cv2.COLOR_GRAY2BGR)
            except:
                pass

            else:
                try:
                    k = img_size/w
                    height_calc = math.ceil(k * h)
                    img_resize = cv2.resize(img_crop, (img_size, height_calc))
                    height_gap = math.ceil((img_size - height_calc)/2)
                    img_white[height_gap:height_gap + height_calc, :] = img_resize
                    #img_white = cv2.Canny(img_white, 50, 100)
                    #img_white = cv2.cvtColor(img_white, cv2.COLOR_GRAY2BGR)
                except:
                    pass
            
        img_white = np.asarray(img_white)
        img_white = (img_white.astype(np.float32) / 127.5) - 1
        data[0] = img_white
        interpreter.set_tensor(input_details[0]['index'], data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        index = np.argmax(output_data)

        # TTS Logic
        words.append(labels[index])
        if len(words) == 4:
            words = words[1:4]

        if len(words) == 3:
            if words[0] == labels[index] and words[1] == labels[index]:
                current_word = labels[index]

                # Play the corresponding sound file
                if current_word in sound_files:
                    sound_file = os.path.join('sounds', sound_files[current_word])
                    playsound.playsound(sound_file)

    else:
        words = []

# Create a separate thread for video capture
def capture_thread():
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            process_frame(frame)
        else:
            break

    cap.release()

# Start the capture thread
capture_thread = Thread(target=capture_thread)
capture_thread.start()
