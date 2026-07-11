import soundcard as sc
import numpy as np


class Microphone:
    def __init__(self, mic_index=0):
        self.mic = sc.all_microphones()[mic_index]
        self.samplerate = 48000
        self.numframes = int(self.samplerate / 2)

    def record_audio(self):
        data = self.mic.record(samplerate=self.samplerate, numframes=self.numframes)
        ch1 = data[:, 0] 
        ch2 = data[:, 1]
        return ch1, ch2
