import numpy as np
import pyaudio
from time import sleep
import speech_recognition
import os
import wave
import io
import tensorflow as tf
import soundfile as sf
import tensorflow_io as tfio
import sounddevice

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
p = pyaudio.PyAudio()

#'''
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

for i in range(0, numdevices):
    print(p.get_device_info_by_host_api_device_index(0, i).get('name'))
    if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print('Input Device id ', i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
#'''

'''
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
micIndex = 0

for i in range (0, numdevices):
    if str((p.get_device_info_by_host_api_device_index(0, i).get('name'))) == 'USB Audio: - (hw:3,0)':
        micIndex = i
        print('micIndex set to: ', p.get_device_info_by_host_api_device_index(0, i).get('name'))
'''

'''
samplerates = 16000, 32000, 44100, 48000, 96000, 128000
device = 2

supported_sr = []
for fs in samplerates:
    try:
        sounddevice.check_output_settings(device=device, samplerate=fs)
    except Exception as e:
        print(fs, e)
    else:
        supported_sr.append(fs)
print(supported_sr)
'''

def record_audio():
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=FRAMES_PER_BUFFER)
    frames = []
    seconds = 1
    for i in range(0, int(RATE/FRAMES_PER_BUFFER * seconds)):
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    return np.frombuffer(b''.join(frames), dtype=np.int16)

def terminate():
    p.terminate