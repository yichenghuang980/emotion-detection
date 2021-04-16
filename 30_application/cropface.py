"""
Command line tool that takes an image, detecs the faces, and crops them
Saves cropped faces to local dir
Usage: python cropface.py --fname {image-filename}
"""

import cv2
import click
import numpy as np
import time


def predict_sentiment(img,model,show_time=False):
	"""
	Predicts the sentiment of a single image
	"""
	face = img.reshape(1,48,48,1) # reshape image from 48x48 to 1x48x48x1 for inference
	emotion_list = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
	start = time.time()
	preds = model.predict(face) # generate prediction array
	end=time.time()
	elapsed = end-start
	#print(f'One prediction made in {elapsed:.4f}s')
	prediction = emotion_list[preds.argmax()] # index emotion list based on index of highest prediction
	return prediction

def detect_face(img, verbose=False):
    """
    Detect faces using the OpenCV haar cascade classifier
    Returns a list of face coordinates
    """
    img = img.astype(np.uint8)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    faces = face_cascade.detectMultiScale(img, 1.1, 4)
    face_cnt = len(faces)
    if verbose == True:
        print(f"Detected faces: {face_cnt}")
    return faces

def box_faces(img,face_dict,sentiments = {}, size=' ',label=True):
    """
    Draws a box around the faces in the image file
    Use the size parameter to set a lower bound for detected face size
    No return
    """
    for i in range(len(face_dict)):
    	text = sentiments[i]
    	face = face_dict[i]
    	(x,y,w,h) = face
    	if type(size) == str:
    		lim = w
    	else:
    		lim = size

    	if w >= lim:
    		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    		if label == True:
    			cv2.putText(img, text, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

def show_image(img_name, img, waitKey=0):
    """
    Shows cv2.imshow() with waitKey(0) and destroyAllWindows
    """
    cv2.imshow(img_name, img)
    cv2.waitKey(waitKey)
    cv2.destroyAllWindows()

def crop(img,face):
    """
    Takes the image and crops the faces
    Returns the cropped face
    """
    height, width = img.shape[:2]
    x, y, w, h = face
    r = max(w, h) / 2
    centerx = x + w / 2
    centery = y + h / 2
    nx = int(centerx - r)
    ny = int(centery - r)
    nr = int(r * 2)
    faceimg = img[ny:ny+nr, nx:nx+nr]
    lastimg = cv2.resize(faceimg, (48, 48))
    return lastimg

def crop_face(img, showimg = False):
    img = img.astype(np.uint8)

    faces = detect_face_alt2(img,verbose=False)

    box_faces(img,faces)

    if showimg == True:
        cv2_imshow(img)


    for face in faces:
        lastimg = crop(img,face)
        if showimg == True:
            cv2_imshow(lastimg)
    return lastimg

@click.command()
@click.option('--fname')
def crop_face_fromfile(fname):
    img = cv2.imread(fname)

    faces = detect_face(img,verbose=True)

    # box_faces(img,faces)

    show_image('img', img)

    face_captures = []
    i=0
    for face in faces:
        lastimg = crop(img,face)
        i += 1
        filename = f'image{i}.jpg'
        face_captures.append(filename)
        cv2.imwrite(filename, lastimg)

    for image in face_captures:
    	face = cv2.imread(image)
    	show_image('face', face)

if __name__ == "__main__":
	crop_face_fromfile()
