import cv2

class VideoCamera(object):
    def __init__(self):
        # capture video from device 0 (webcam)
        self.video = cv2.VideoCapture(-1)
        #self.frame = self.get_frame()
        
    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        # opencv uses raw images by default, encode into jpg to display it
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    # def draw_box(self, x1, x2, y1, y2, colour):
    #     cv2.rectangle(self.frame, (x1, x2), (y1, y2), colour)
        

