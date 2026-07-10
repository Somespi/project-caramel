import cv2


class Camera:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)

        if not self.cam.isOpened():
            print("Error: Could not open the webcam.")
            exit()

        self.frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def read_frame(self):
        ret, frame = self.cam.read()
        if not ret:
            print("Error: Failed to grab frame.")
            return None
        return frame
    
    def release(self):
        self.cam.release()
        cv2.destroyAllWindows()