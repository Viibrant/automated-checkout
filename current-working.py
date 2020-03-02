import numpy as np
import cv2
import keras
from keras.preprocessing.image import img_to_array
from keras.applications import mobilenet
from keras.preprocessing.image import load_img
from keras.applications.imagenet_utils import decode_predictions
from PIL import Image
from numpy import asarray
import process
cap = cv2.VideoCapture(0)
model = mobilenet.MobileNet(weights='imagenet')



while(True):
    ret, frame = cap.read()
    # get the image and convert it into a numpy array and into a format for our model
    img = Image.fromarray(frame, 'RGB')
    img = img.resize((224,224), Image.ANTIALIAS)
    np_image = img_to_array(img)
    image_batch = np.expand_dims(np_image, axis=0)
    processed_image = mobilenet.preprocess_input(image_batch.copy())
    
    # actual machine learning part
    predictions = model.predict(processed_image)
    labels = decode_predictions(predictions)
    for label in labels:
        print("Prediction: %s, Probability: %s"%(label[0][1], label[0][2]))
 
    if ret == True:
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('l'):
            break        
    else:
        break

cap.release()
cv2.destroyAllWindows()
