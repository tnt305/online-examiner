from typing import Dict
import torch 
import base64
import numpy as np


from pyannote.audio import Pipeline
from pyannote.core import Annotation

import speech_recognition as sr
import pyaudio
import wave
import time
import threading
import os

def save_audio(stream, filename):
    """
    Saving the recorded audio sample in record.
    """
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt32  # 16 bits per sample
    channels = 2
    fs = 44100  # Record at 44100 samples per second
    seconds = 10  # Number of seconds to record at once
    
    
    path = "./audio_logs/audios/"
    filename = path + filename
    
    frames = []  # Initialize array to store frames
    
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
    
    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    # Stop and close the stream
    stream.stop_stream()
    stream.close()

class AudioTracker():
    def __init__(self, model_path, hugging_face_token = None, sample_rate = 16000):
        # load the model
        self.hugging_face_token = hugging_face_token
        self.sample_rate =sample_rate
        self.pipeline = Pipeline.from_pretrained(model_path, use_auth_token= self.hugging_face_token)

    def __call__(self, data: Dict[str, bytes]) -> Dict[str, str]:
        """
        Args:
            data (:obj:):
                includes the deserialized audio file as bytes
        Return:
            A :obj:`dict`:. base64 encoded image
        """
        # process input
        # retrieve input if exists 
        # calls out during the inference
        inputs = data.pop("inputs", data)
        parameters = data.pop("parameters", None) #  min_speakers=2, max_speakers=5

        # decode the base64 audio data
        audio_data = base64.b64decode(inputs)
        audio_nparray = np.frombuffer(audio_data, dtype=np.int16)

        # prepare pynannote input
        audio_tensor= torch.from_numpy(audio_nparray).float().unsqueeze(0)
        pyannote_input = {"waveform": audio_tensor, "sample_rate": self.sample_rate}
        
        # apply pretrained pipeline
        # pass inputs with all kwargs in data
        if parameters is not None:
            diarization = self.pipeline(pyannote_input, **parameters)
        else:
            diarization = self.pipeline(pyannote_input)

        # postprocess the prediction
        processed_diarization = [
            {"label": str(label), "start": str(segment.start), "stop": str(segment.end)}
            for segment, _, label in diarization.itertracks(yield_label=True)
        ]
        
        return processed_diarization
