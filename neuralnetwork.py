from imageai.Detection import ObjectDetection
from PIL import Image
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf
import numpy as np
import io
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
class NeuralNetwork(object):
    def __init__(self):
        self.model = ObjectDetection()
        self.model.setModelTypeAsYOLOv3()
        self.model.setModelPath(os.path.join(os.getcwd(), "yolo.h5"))
        self.model.loadModel(detection_speed="faster")
    
    def predict(self, raw):
        print("Predicting...")
        image = np.array(Image.open(io.BytesIO(raw)))
        detections = self.model.detectObjectsFromImage(input_image=image, input_type="array", output_type="array", minimum_percentage_probability=30)
        print(detections[1])
        return detections[1]

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)