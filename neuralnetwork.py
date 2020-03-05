import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import mobilenet
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from PIL import Image
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