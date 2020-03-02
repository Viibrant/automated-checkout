import cv2

class VideoCamera(object):
    def __init__(self):
        # capture video from device 0 (webcam)
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        # opencv uses raw images by default, encode into jpg to display it
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
