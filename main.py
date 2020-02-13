import numpy as np
import cv2
from PIL import Image

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    img = Image.fromarray(frame, 'RGB')
    print(img)
    if ret == True:
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('l'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()