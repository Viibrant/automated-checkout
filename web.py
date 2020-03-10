from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit

from camera import VideoCamera
from neuralnetwork import NeuralNetwork

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

####### App Routes ########
@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    global frame
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict')
def predict():
    global frame
    model = NeuralNetwork()
    print("Prediction Initialising.\n.\n.\n.\n.\n.")

    while True:
        predictions = model.predict(frame)
        print(predictions)
        for prediction in predictions:
            socketio.emit ('predict', {'data': "Prediction: %s, Probability: %s\n" 
                % (prediction[0][1], prediction[0][2])})
    
####### Socket Events ########
@socketio.on('my event')                          
def test_message(message):
    for x in range(500):    
        emit('my response', {'data': 'got it %s times!'%(x)})

if __name__ == '__main__':
    app.debug = True
    socketio.run(app, host='0.0.0.0') 