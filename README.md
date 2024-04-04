# ⚠️ Archive notice 📚 2024/04
This repo contains my A Level Computer Science coursework.

>*🔧 I used [YOLOv3](https://pjreddie.com/darknet/yolo/), then SOTA pretrained object detector, for an automated checkout. It served as an initial foray into applying neural networks within a practical, albeit simplified, retail setting. This project remains as both a record of my early exploration into machine learning, yet also as simple nostalgia.*

>*🧠 I learnt a lot from this project. The bit I was particularly proud of was getting the frontend websocket code to work, pulling data streamed from the backend and dynamically creating a table with the detected objects. It was my first time working with generator functions, which I hadn’t even heard of before diving into this. Also, this was my first shot at putting together a system that used different programming languages, which I found quite exciting.*

---

# Automated-Checkout
Main repository for this project, built using Flask, ImageAI.

## Execution
To run (pretty obvious but still):
```bash
./start.sh
```
This will start a Flask web server run locally on your computer that you can connect to, the URL should be outputted to the terminal.

Project consists of **web.py**, a script that runs the web server itself. Here we find the code instantiating the camera declared in camera.py and the Neural Network declared in, you guessed it, **neuralnetwork.py**. 

## Dependencies
```bash
pip install tensorflow opencv keras flask flask_socketio imageai pillow numpy
```
Basically you may as well just install Anaconda that'll probably sort you all out considering
