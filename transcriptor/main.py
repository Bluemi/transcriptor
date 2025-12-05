#!/usr/bin/env python3

from typing import Optional

import numpy as np
import sounddevice as sd
import pygame as pg

from peng_ui.elements import TextField, Label
from peng_ui.utils import Align
from peng_ui.viewer import Viewer

from transcriptor.whisper import WhisperDecoder

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
        super().__init__(screen_size=(0, 0), flags=pg.FULLSCREEN)
        # super().__init__(screen_size=(800, 600))
        screen_width, screen_height = self.screen.get_size()

        self.device_info = sd.query_devices()
        print(self.device_info)
        self.device_index = sd.default.device[0]

        label_width = 120
        self.label = Label(
            pg.Rect((screen_width - label_width) // 2, screen_height-60, label_width, 40), "Press Space to record"
        )

        text_field_width = screen_width - 600
        text_field_height = screen_height - 50 - 80
        self.text_field = TextField(
            pg.Rect(30, 30, text_field_width, text_field_height), "Recorded text"
        )

        self.audio_device_label = Label(
            pg.Rect(text_field_width + 30 + 50, 60, 120, 40), f"Audio device: ", align=Align.LEFT
        )
        self.audio_device_label2 = Label(
            pg.Rect(text_field_width + 30 + 50, 110, 120, 40),
            f"{self.device_index} - {self.device_info[self.device_index]['name']}", align=Align.LEFT
        )
        self.audio_device_label3 = Label(
            pg.Rect(text_field_width + 30 + 50, 160, 120, 40), f"Use Up/Down to change audio device ", align=Align.LEFT
        )

        self.input_stream: Optional[sd.InputStream] = None
        self.buffer = AudioBuffer()

        self.whisper = WhisperDecoder()


    def handle_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
                break
            elif event.type == pg.KEYDOWN and event.key == pg.K_SPACE and not self.text_field.is_focused:
                self.label.set_text('Recording...')
                self.input_stream = sd.InputStream(
                    samplerate=SAMPLERATE, callback=self.buffer, device=self.device_index, dtype='int16'
                )
                self.input_stream.start()
            elif event.type == pg.KEYUP and event.key == pg.K_SPACE:
                self.transcribe_audio()
            elif event.type == pg.KEYDOWN and event.key == pg.K_DOWN:
                if self.device_index > 0:
                    self.device_index -= 1
                    self.audio_device_label2.set_text(f"{self.device_index} - {self.device_info[self.device_index]['name']}")
            elif event.type == pg.KEYDOWN and event.key == pg.K_UP:
                if self.device_index < len(self.device_info) - 1:
                    self.device_index += 1
                    self.audio_device_label2.set_text(f"{self.device_index} - {self.device_info[self.device_index]['name']}")

            for elem in self.iter_elements():
                elem.handle_event(event)

    def transcribe_audio(self):
        if self.input_stream is not None:
            self.label.set_text('Press Space to record')
            self.input_stream.stop()
            self.input_stream.close()
            recording = self.buffer.get()
            if len(recording) == 0:
                print('No audio recorded')
            else:
                print(f'recorded {recording.shape[0] / SAMPLERATE} seconds')
                res_text = self.whisper(recording)
                self.text_field.set_text(res_text)
            self.buffer.clear()
            self.input_stream = None


if __name__ == '__main__':
    viewer = MyViewer()
    viewer.run()
