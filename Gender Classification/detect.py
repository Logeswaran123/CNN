# Detection

from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
import numpy as np
import cv2
import os
import time
import cvlib as cv

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Read video
video = cv2.VideoCapture("videoplayback.mp4")
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('out.mp4', fourcc, 60, (320, 240))

model_path = 'gender_detection.model'
# Load pre-trained model
model = load_model(model_path)

# Exit if video not opened
if not video.isOpened():
    print("Could not open video")

# Read first frame
ok, frame = video.read()
if not ok:
    print('Cannot read video file')

while True:
    
    
    start_time = time.time()
    ok, frame = video.read()
    if not ok:
        break
    
    
    
    # Detect faces in the image
    face, confidence = cv.detect_face(frame)
    
    classes = ['man','woman']
    
    # Loop through detected faces
    for idx, f in enumerate(face):
    
         # Get corner points of face rectangle       
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
    
        # Draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
    
        # Crop the detected face region
        face_crop = np.copy(frame[startY:endY,startX:endX])
    
        # Preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)
    
        # Apply gender detection on detected face
        conf = model.predict(face_crop)[0]
        print(conf)
        print(classes)
        
        
        # Get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]
    
        label = "{}: {:.2f}%".format(label, conf[idx] * 100)
    
        Y = startY - 10 if startY - 10 > 10 else startY + 10
    
        # Write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
    # Display output
    cv2.imshow("Output", frame)
    
    # Break if ESC pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break
    print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop



# Save output
#cv2.imwrite("Ouput_file.jpg", image)

video.release()
cv2.destroyAllWindows()
