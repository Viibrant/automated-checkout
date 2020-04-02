from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from camera import VideoCamera
from neuralnetwork import NeuralNetwork

app = Flask(__name__)
socketio = SocketIO(app)
v = VideoCamera()
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
    global v
    return Response(gen(v),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict')
def predict():
    print("Prediction Initialising.\n.\n.\n.\n.\n.")
    global frame
    global v
    model = NeuralNetwork()
    #v.draw_box(0, 0, 69, 69, (255, 0, 0))
    while True:
        predictions = model.predict(frame)
        #for prediction in predictions:
            #socketio.emit ('predict', {'data': "Prediction:<font color='red'>%s</font>\nProbability: <font color='red'>%s<font>"
            #    % (prediction[0][1], prediction[0][2])})
        
        for eachObject in predictions:
            print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
            print("--------------------------------")

        socketio.emit('predict', {'data': predictions})

####### Socket Events ########
@socketio.on('my event')                          
def test_message(message):
    for x in range(500):
        emit('my response', {'data': 'got it %s times!'%(x)})

if __name__ == '__main__':
    app.debug = True
    app.passthrough_errors = True
    socketio.run(app, host='0.0.0.0')
