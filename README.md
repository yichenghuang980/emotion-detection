# 1. Introduction

The goal of this project is to present a final application that can help content presenters to evaluate the overall performance of the conference and improve communication efficiency. 

Specifically, this application detects real time facial expressions/engagement of the audience using a convolutional neural network in a video conference setting and assesses the overall sentiment/engagement. 

For more information, please refer to the [Full Report](https://github.com/yichenghuang980/emotion-detection/blob/master/Final_Report.pdf)

# 2. Demo Video

[![](http://img.youtube.com/vi/atmCf3voXn4/0.jpg)](https://youtu.be/atmCf3voXn4)

# 3. Setup Instruction

Step 1: Clone this repository

```
git clone https://github.com/yichenghuang980/emotion-detection.git
```

Step 2: Change directory into the project folder

```
cd emotion-detection
```

Step 3:Create virtual environment if none exists (optional) 

```
virtualenv --python $(which python3) venv
source venv/bin/activate
```

Step 4: Change directory into the application folder

```
cd 30_application
```

Step 5: Install all the required packages 

```
pip install -r requirements.txt
```

Step 6: Run the following command to crop faces with openCV and save them in your local drive

```
python cropface.py --fname {image-filename}
```

Step 7: Run the following command to genereate real-time video yourselves

```
python screenface.py
```
