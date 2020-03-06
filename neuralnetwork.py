import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import mobilenet
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from PIL import Image
import tensorflow as tf
import io

class NeuralNetwork(object):
    def __init__(self):
        self.model = mobilenet.MobileNet(weights='imagenet')
    
    def predict(self, frame):
        # get the image and convert it into a numpy array and into a format for our model
        #img = Image.fromarray(frame, 'RGB')
        img = Image.open(io.BytesIO(frame))
        img = img.resize((224,224), Image.ANTIALIAS)
        np_image = img_to_array(img)
        image_batch = np.expand_dims(np_image, axis=0)
        processed_image = mobilenet.preprocess_input(image_batch.copy())
        
        # actual machine learning part
        predictions = self.model.predict(processed_image)
        return decode_predictions(predictions)

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