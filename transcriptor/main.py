#!/usr/bin/env python3
from typing import Optional

from scipy.io.wavfile import write
import numpy as np
import sounddevice as sd
import pygame as pg

from peng_ui.elements import TextField, Label
from peng_ui.viewer import Viewer

SAMPLERATE = 16000


class AudioBuffer:
    def __init__(self):
        self.buffer = []

    def __call__(self, indata: np.ndarray, frames: int, time, status: sd.CallbackFlags):
        self.buffer.append(indata[:, 0].copy())

    def clear(self):
        self.buffer = []

    def get(self) -> np.ndarray:
        if not self.buffer:
            return np.array([])
        return np.concatenate(self.buffer).flatten()


class MyViewer(Viewer):
    def __init__(self):
        # super().__init__(screen_size=(0, 0), flags=pg.FULLSCREEN)
        super().__init__(screen_size=(800, 600))
        screen_width, screen_height = self.screen.get_size()

        label_width = 120
        self.label = Label(
            pg.Rect((screen_width - label_width) // 2, screen_height-60, label_width, 40), "Press Space to record"
        )

        text_field_width = screen_width - 200
        text_field_height = screen_height - 50 - 80
        self.text_field = TextField(
            pg.Rect((screen_width - text_field_width) // 2, 50, text_field_width, text_field_height), "Recorded text"
        )

        self.input_stream: Optional[sd.InputStream] = None
        self.buffer = AudioBuffer()

        self.device_info = sd.query_devices()
        print(type(self.device_info))

    def handle_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
                break
            elif event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                self.label.set_text('Recording...')
                self.input_stream = sd.InputStream(samplerate=SAMPLERATE, callback=self.buffer, device=9, dtype='int16')
                self.input_stream.start()
            elif event.type == pg.KEYUP and event.key == pg.K_SPACE:
                self.label.set_text('Press Space to record')
                self.input_stream.stop()
                self.input_stream.close()
                recording = self.buffer.get()
                print(f'recorded {recording.shape[0] / SAMPLERATE} seconds')

                write('recording.wav', SAMPLERATE, recording)
                self.buffer.clear()
                self.input_stream = None

            for elem in self.iter_elements():
                elem.handle_event(event)


if __name__ == '__main__':
    viewer = MyViewer()
    viewer.run()
