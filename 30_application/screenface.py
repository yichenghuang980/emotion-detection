"""
Takes a user screen-capture video feed, and detects faces
**Uses fuctions from cropface.py***
"""

import cv2 
import numpy as np
from PIL import ImageGrab
from cropface import detect_face, box_faces, crop, predict_sentiment
import tensorflow as tf
import time

model = tf.keras.models.load_model('../40_models/emotion-detection-model-A-opencv.h5')

while True:
	cap = ImageGrab.grab()
	start = time.time()
	cap_np = np.array(cap)
	
	frame = cv2.cvtColor(cap_np,cv2.COLOR_BGR2RGB)
	gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
	faces = detect_face(gray)

	croped = [crop(gray, face) for face in faces]
	sentiments = []
	for face in croped:
		sentiments.append(predict_sentiment(face,model))

	box_faces(frame,faces,sentiments,size=0)
	print(f"{len(faces)} faces detected.")
	sent_dict = {x:sentiments.count(x) for x in sentiments}
	print(sent_dict)

	resize = cv2.resize(frame, (854,480))
	end=time.time()
	elapsed = end-start
	print(f'Time elapsed: {elapsed :.4f}s')
	cv2.imshow('frame',resize)
	if cv2.waitKey(1) == 27:
		break

cv2.destroyAllWindows() 