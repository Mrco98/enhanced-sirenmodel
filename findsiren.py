import os
os.environ["tf_gpu_allocator"]="cuda_malloc_async"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from tensorflow.keras import models
from tensorflow import keras
from recording_helper import record_audio, terminate
from predict import make_prediction
from matplotlib import pyplot as plt
import tensorflow as tf

loaded_model = keras.models.load_model('models/lstm.h5')

def preprocess():
    pass

def predict_mic():
    detected = False
    audio = record_audio()
    data = preprocess(audio)
    prediction = loaded_model(data)
    if prediction >= 0.8:
        print("=====SIREN DETECTED=====")
        print(f'Confidence: {prediction}')
        detected = True
    else:
        print("=====LISTENING=====")
        detected = False
    
    return detected, prediction

if __name__ == "__main__":
    while True:
        predict_mic()