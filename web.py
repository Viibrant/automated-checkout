from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from camera import VideoCamera
from neuralnetwork import NeuralNetwork
import numpy
import json

app = Flask(__name__)
socketio = SocketIO(app)
v = VideoCamera()
####### App Routes ########
@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(v),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def predict(frame):
    print("Prediction Initialising.\n.\n.\n.\n.\n.")
    model = NeuralNetwork()
    predictions = model.predict(frame)
    def convert(o):
        if isinstance(o, numpy.int32): return int(o) 
        raise TypeError

    socketio.emit('predict', {'data': json.dumps(predictions, default=convert)})

####### Socket Events ########
@socketio.on('picture')                          
def snap(picture):
    image = v.get_frame()
    predict(image)

if __name__ == '__main__':
    app.debug = True
    app.passthrough_errors = True
    socketio.run(app, host='0.0.0.0')
