from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO, emit
from camera import VideoCamera
from neuralnetwork import NeuralNetwork
from collections import Counter
import numpy
import json

app = Flask(__name__)
socketio = SocketIO(app)
v = VideoCamera()
####### App Routes ########
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(v),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def predict(frame):
    print("Prediction Initialising.\n.\n.\n.\n.\n.")
    model = NeuralNetwork()
    predictions = model.predict(frame)

    quantities = Counter(prediction['name'] for prediction in predictions)
    products = []
    with open('prices.json') as json_file:
        for key, element in quantities.items():
            try: 
                data = json.load(json_file)
                products.append({
                    'product': key,
                    'quantity': element,
                    'price': data[key] * element
                })
            except:
                products.append({
                    'product': key,
                    'quantity': element,
                    'price': 0
                })
    print(products)
    socketio.emit('predict', {'data': json.dumps(products)})

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

####### Socket Events ########
@socketio.on('picture')                          
def snap(picture):
    image = v.get_frame()
    predict(image)

if __name__ == '__main__':
    app.debug = True
    app.passthrough_errors = True
    socketio.run(app, host='0.0.0.0')
