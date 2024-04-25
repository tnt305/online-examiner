import os
import threading
import wave

import numpy as np
import pyaudio
import keyboard

from audio_tracker import AudioTracker

# Global flag to control recording
stop_recording = False

def save_audio(stream, filename):
    """
    Saving the recorded audio sample in record.
    """
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # Use paInt16 instead of paInt32
    channels = 2
    fs = 44100  # Record at 44100 samples per second
    seconds = 30  # Number of seconds to record at once

    frames = []  # Initialize array to store frames
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    amplify_audio(filename, amplification_factor=2.0)

def amplify_audio(filename, amplification_factor):
    """
    Amplify the audio signal in the specified WAV file.
    """
    # Read audio data from file
    with wave.open(filename, 'rb') as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    # Convert audio data to numpy array
    audio_data = np.frombuffer(frames, dtype=np.int16)

    # Amplify audio data
    amplified_data = (audio_data * amplification_factor).astype(np.int16)

    # Write amplified audio data back to file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        wf.writeframes(amplified_data.tobytes())

    print("Audio amplified successfully")

def record_audio():
    """
    Continuously records 30-second audio segments and saves them.
    """
    global stop_recording
    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # Use paInt16 instead of paInt32
    channels = 2
    fs = 44100

    i = 0
    while not stop_recording:
        print("Monitoring candidate at ", i + 1)
        filename = 'record' + str(i) + '.wav'
        path = "./audio_logs/audios/"
        os.makedirs(path, exist_ok=True)

        sound = os.path.join(path, filename)
        stream = p.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True)
        
        save_audio(stream, sound)
        #audio_detector = AudioTracker(model_path = "pyannote/speaker-diarization-3.1", hugging_face_token = None)

        i += 1

    p.terminate()

# # Start recording in a separate thread
# recording_thread = threading.Thread(target=record_audio)
# recording_thread.start()

# # Wait for the user to press 'q' to stop recording
# keyboard.wait('q')

# # Set the stop flag to True to stop the recording loop
# stop_recording = True

# # Wait for the recording thread to finish
# recording_thread.join()
