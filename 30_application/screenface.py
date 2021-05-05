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
# model = tf.keras.models.load_model('../40_models/model_ResNet.h5')
frame_trac = 0
sent_log = {'Angry':0,'Disgust':0,'Fear':0,'Happy':0,'Sad':0,'Surprise':0,'Neutral':0}

while True:
	frame_trac += 1
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
	sent_dict = {x:sentiments.count(x) for x in sentiments}

	end=time.time()
	elapsed = end-start
	# print(f'Frame {frame_trac}, time elapsed: {elapsed :.4f}s')
	# Get the 6 frame-average sentiment
	for x in sent_dict.keys():
		sent_log[x] += sent_dict[x]

	if frame_trac%2 == 0:
		for x in sent_log.keys():
			sent_log[x] /= 2
			sent_log[x] = round(sent_log[x],3)
		print('\n\n')
		print(f"{len(faces)} faces detected.")
		for item in sent_log.items():
			print(item)
		print('\n\n')
		

	resize = cv2.resize(frame, (854,480))
	cv2.imshow('frame',resize)
	if cv2.waitKey(1) == 27:
		break

cv2.destroyAllWindows() 
