# emotion-detection

Facial Expression Recognition and Emotion Detection in Remote Learning Condition

## To recreate this project

First clone this repository

```
git clone https://github.com/yichenghuang980/emotion-detection.git
```

change directory into the project folder

```
cd emotion-detection
```

(optional) create virtual environment if none exists

```
virtualenv --python $(which python3) venv
source venv/bin/activate
```

change into the application folder

```
cd 30_application
```

install all the required packages 

```
pip install -r requirements.txt
```

run the following command to crop faces with openCV and save them in your local drive

```
python cropface.py --fname {image-filename}
```

run the following command to genereate real-time video yourselves

```
python screenface.py
```
