import os
import shutil
import time
import cv2
import torch
import threading 
from . import camera, latent

CAPTURE_INTERVAL_SECS = 2.0    
INPUT_DIM = 64 * 64 * 3  
TEMP_DIR = "temp_frames"
MODEL_SAVE_PATH = "caramel_visual.pth"

class Eyes:
    def __init__(self, train_epochs=10):  
        self.TRAIN_EPOCHS = train_epochs
        self.cam = camera.Camera()
        self.model = latent.Model(INPUT_DIM)
        self.is_sleeping = False
        self.frame = None
        import numpy as np
        self.current_latent = np.zeros(18, dtype=np.float32)
        
        if os.path.exists(MODEL_SAVE_PATH):
            try:
                self.model.load(MODEL_SAVE_PATH)
            except Exception as e:
                ...

    def __start(self):
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)
        frame_count = 0
        last_capture_time = 0  
        while not self.is_sleeping:
            current_time = time.time()
            if current_time - last_capture_time >= CAPTURE_INTERVAL_SECS:
                frame = self.cam.read_frame()
                if frame is not None:
                    frame_count += 1
                    frame_resized = cv2.resize(frame, (64, 64))
                    self.frame = frame_resized
                    self.current_latent = self.model.generate_latent(frame_resized)
                    frame_filename = os.path.join(TEMP_DIR, f"frame_{frame_count:04d}.png")
                    cv2.imwrite(frame_filename, frame_resized)
                    
                last_capture_time = current_time
                
            time.sleep(0.1)  

    def run(self):
        self.is_sleeping = False
        self.thread = threading.Thread(target=self.__start, daemon=True) 
        self.thread.start()
        
    def get_frame(self):
        return self.current_latent, self.frame
        
    def stop(self):
        self.is_sleeping = True
        if hasattr(self.thread, 'join'):
            self.thread.join() 
        try:
            self.cam.release()
        except AttributeError:
            pass
    
    def train(self):
        if not os.path.exists(TEMP_DIR) or not os.listdir(TEMP_DIR):
            return

        self.model.train_from_frames_path(TEMP_DIR, self.TRAIN_EPOCHS)
        self.model.save(MODEL_SAVE_PATH)
        shutil.rmtree(TEMP_DIR)
