import os
import shutil
import time
import torch
import threading
import uuid
import pickle
import numpy as np

from . import microphone, latent

CAPTURE_INTERVAL_SECS = 2
TEMP_DIR = "temp_segments"
MODEL_SAVE_PATH = "caramel_audio.pth"

mic = microphone.Microphone()
model = latent.Model(input_length=mic.numframes) # Map to input_length initialization parameter

class Ear:
    def __init__(self, train_epochs=50):
        self.model = model
        self.mic = mic
        self.epochs = train_epochs

        self.recording = False
        self.thread = None
        self.audio_data = None

        # CNN-LSTM latent space features (matching size 128)
        self.current_latent = np.zeros(128, dtype=np.float32)

        if os.path.exists(MODEL_SAVE_PATH):
            try:
                self.model.load(MODEL_SAVE_PATH)
                print("Loaded audio cortex model weights successfully.")
            except Exception as e:
                print("Model loading failed:", e)

    def __start(self):
        os.makedirs(TEMP_DIR, exist_ok=True)
        last_capture_time = 0

        while self.recording:
            current_time = time.time()
            if current_time - last_capture_time >= CAPTURE_INTERVAL_SECS:
                waveform, _ = self.mic.record_audio()

                if waveform is not None:
                    self.audio_data = waveform

                    # 💡 Fixed: model.generate_latent already yields a flat numpy array
                    latent_vector = self.model.generate_latent(waveform)
                    self.current_latent = latent_vector

                    # Serialize the raw audio waveform segment for sleep training
                    filename = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.pkl")
                    with open(filename, "wb") as f:
                        pickle.dump(waveform, f)

                last_capture_time = current_time

            time.sleep(0.01)

    def run(self):
        if self.recording:
            return
        self.recording = True
        self.thread = threading.Thread(target=self.__start, daemon=True)
        self.thread.start()

    def get_frame(self):
        """Returns the current size-128 audio latent vector alongside raw sound metrics"""
        return self.current_latent, self.audio_data

    def stop(self):
        self.recording = False
        if self.thread is not None:
            self.thread.join()

    def train(self):
        if not os.path.exists(TEMP_DIR) or len(os.listdir(TEMP_DIR)) == 0:
            print("No training data gathered for this audio sleep cycle.")
            return

        print(f"Entering audio sleep training loop for {self.epochs} epochs...")
        self.model.train_from_frames_path(TEMP_DIR, self.epochs)
        self.model.save(MODEL_SAVE_PATH)
        shutil.rmtree(TEMP_DIR)
        print("Audio sleep cycles finished. Raw waveforms purged.")