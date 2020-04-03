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
