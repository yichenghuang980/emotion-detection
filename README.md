# 1. Introduction


The goal of this project is to present a final application that can help content presenters to evaluate the overall performance of the conference and improve communication efficiency. Specifically, this application detects real time facial expressions/engagement of the audience using a convolutional neural network in a video conference setting and assesses the overall sentiment/engagement. For this progress report, the goal is to evaluate the performance of different CNN architectures to detect facial expression.


# 2. Demo Video

[![](http://img.youtube.com/vi/atmCf3voXn4/0.jpg)](https://youtu.be/atmCf3voXn4)

# 3. Setup Instruction

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
