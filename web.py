from flask import Flask, render_template, Response
from camera import VideoCamera
from neuralnetwork import NeuralNetwork

app = Flask(__name__)


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
    model = NeuralNetwork()
    def pred(model):
        global frame
        while True:
            predictions = model.predict(frame)
            for prediction in predictions:
                yield ("Prediction: %s, Probability: %s\n" 
                    % (prediction[0][1], prediction[0][2]))
    print("Started Predicting.\n.\n.\n.\n.\n.")
    return Response(pred(model),mimetype='text/plain')
        

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', threaded=True)
