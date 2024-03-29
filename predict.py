from tensorflow.keras.models import load_model
from clean import downsample_mono, envelope
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
import numpy as np
from glob import glob
import argparse
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt


def make_prediction(args):

    model = load_model(args.model_fn,
        custom_objects={'STFT':STFT,
                        'Magnitude':Magnitude,
                        'ApplyFilterbank':ApplyFilterbank,
                        'MagnitudeToDecibel':MagnitudeToDecibel})
    wav_paths = glob('{}/**'.format(args.src_dir), recursive=True)
    wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths if '.wav' in x])
    classes = sorted(os.listdir(args.src_dir))
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    le = LabelEncoder()
    y_true = le.fit_transform(labels)
    results = []
    
    fn_prob = {}

    for z, wav_fn in tqdm(enumerate(wav_paths), total=len(wav_paths)):
        rate, wav = downsample_mono(wav_fn, args.sr)
        mask, env = envelope(wav, rate, threshold=args.threshold)
        clean_wav = wav[mask]
        step = int(args.sr*args.dt)
        batch = []

        for i in range(0, clean_wav.shape[0], step):
            sample = clean_wav[i:i+step]
            sample = sample.reshape(-1, 1)
            if sample.shape[0] < step:
                tmp = np.zeros(shape=(step, 1), dtype=np.float32)
                tmp[:sample.shape[0],:] = sample.flatten().reshape(-1, 1)
                sample = tmp
            batch.append(sample)
        X_batch = np.array(batch, dtype=np.float32)
        y_pred = model.predict(X_batch, verbose=0)
        y_mean = np.mean(y_pred, axis=0)
        y_pred = np.argmax(y_mean)
        y_predictions.append(y_pred)
        real_class = os.path.dirname(wav_fn).split('/')[-1]
        y_truee.append(classes.index(real_class))
        print('File Name: '+wav_fn, 'Actual class: {}, Predicted class: {}'.format(real_class, classes[y_pred]))
        results.append(y_mean)

    np.save(os.path.join('logs', args.pred_fn), np.array(results))
    return y_truee, y_predictions
    

 
 
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--model_fn', type=str, default='models/conv2d.h5',
                        help='model file to make predictions')
    parser.add_argument('--pred_fn', type=str, default='T1',
                        help='fn to write predictions in logs dir')
    parser.add_argument('--src_dir', type=str, default='D:/SirenNeuralNetwork/enhanced-sirenmodel/validation/T1',
                        help='directory containing wavfiles to predict')
    parser.add_argument('--dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='sample rate of clean audio')
    parser.add_argument('--threshold', type=str, default=20,
                        help='threshold magnitude for np.int16 dtype')
    args, _ = parser.parse_known_args()

    y_predictions = []
    y_truee = []
    make_prediction(args)

    #ACCURACY
    acc_score = accuracy_score(y_true=y_truee, y_pred=y_predictions)
    print(f'Accuracy: {acc_score:.3f}')

    #PRECISION
    precision = precision_score(y_true=y_truee, y_pred=y_predictions, average='micro')
    print(f"Precision: {precision:.3f}")

    #RECALL
    recall = recall_score(y_truee, y_predictions, average='micro')
    print(f"Recall: {recall:.3f}")

    #F1 SCORE
    f1 = f1_score(y_true=y_truee, y_pred=y_predictions, average='micro')
    print(f"F1 Score: {f1:.3f}")

    print(y_truee, len(y_truee))
    print(y_predictions, len(y_predictions))

    cm = confusion_matrix(y_truee, y_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Siren', 'Traffic'], yticklabels=['Siren', 'Traffic'])
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix - Gunshot')
    plt.show()

