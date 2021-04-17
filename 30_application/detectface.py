"""
Takes a user webcam video feed, and detects faces
**Uses fuctions from cropface.py***
"""

import cv2 
import numpy as np
from cropface import detect_face, box_faces, crop, predict_sentiment
import time
import tensorflow as tf

cap = cv2.VideoCapture(0)
model = tf.keras.models.load_model('../40_models/emotion-detection-model-A-opencv.h5')

while True:
	ret, frame = cap.read() # Capture the video feed frame
	print('New frame')
	start = time.time()

	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # convert to gray scale (face detection preprocessing)

	faces = detect_face(gray,verbose=True) # detect faces (returns list of face coordinates)

	face_dict = {i:face for (i,face) in enumerate(faces)} # Create a dictionary mapping for each face

	croped = {i:crop(gray, face) for (i,face) in face_dict.items()} # Create a dictionary mapping for each face key to a cropped face

	sentiments = {} # Initialize dictionary mapping face to sentiment
	for i in range(len(croped)):
		sentiments[i] = predict_sentiment(croped[i],model) # Predict sentiment and map prediction to face key

	box_faces(frame,face_dict,sentiments,60) # Draw boxes and sentiment on original video frame

	end = time.time()
	elapsed = end-start
	print(f'Time elapsed {elapsed :.4f}s\n')

	cv2.imshow('frame',frame)
	if cv2.waitKey(1) == 27:
		break

cap.release()
cv2.destroyAllWindows() 