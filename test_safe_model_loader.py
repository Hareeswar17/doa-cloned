
# -*- coding: utf-8 -*-
""" DOA test script using safely loaded .keras model """

import csv
import os
import argparse
import math
import grid
import dsp
import numpy as np
from scipy.io.wavfile import read
from feature_extractor import getNormalizedIntensity

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout,
                                     Reshape, Bidirectional, LSTM, TimeDistributed, Dense)
from keras.saving import get_custom_objects

def build_cartesian_model(input_shape=(25, 513, 6)):
    inp = Input(shape=input_shape, name="input0")
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name="conv00")(inp)
    x = BatchNormalization(axis=1, name="norm00")(x)
    x = MaxPooling2D(pool_size=(1, 8), strides=(1, 8), name="pool00")(x)
    x = Dropout(0.2, name="dropout00")(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name="conv01")(x)
    x = BatchNormalization(axis=1, name="norm01")(x)
    x = MaxPooling2D(pool_size=(1, 8), strides=(1, 8), name="pool01")(x)
    x = Dropout(0.2, name="dropout01")(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name="conv02")(x)
    x = BatchNormalization(axis=1, name="norm02")(x)
    x = MaxPooling2D(pool_size=(1, 4), strides=(1, 4), name="pool02")(x)
    x = Dropout(0.2, name="dropout02")(x)

    x = Reshape((25, -1), name="reshape00")(x)
    x = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.2), name="lstm00")(x)
    x = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.2), name="lstm01")(x)

    x = TimeDistributed(Dense(429, activation='linear'), name="td00")(x)
    x = Dropout(0.2, name="dropout05")(x)
    out = TimeDistributed(Dense(3, activation='linear'), name="doa_out")(x)

    return Model(inp, out)

def toLOCATA(azi, ele):
    ele = 90 - ele
    return azi, ele

def tensor_angle(a, b):
    half_delta = (a - b) / 2
    temp = np.clip(math.sin(half_delta[..., 1]) ** 2 + math.sin(a[..., 1]) * math.sin(b[..., 1]) * math.sin(
        half_delta[..., 0]) ** 2, 1e-9, None)
    angle = 2 * math.asin(np.clip(math.sqrt(temp), None, 1.0 - 1e-9))
    num_nan = np.sum(np.isnan(angle))
    if num_nan > 0:
        print("encountered {} NANs".format(num_nan))
    return angle

def acn2fuma(x_in):
    x_out = np.array([x_in[0, :] / math.sqrt(2.0), x_in[3, :], x_in[1, :], x_in[2, :]])
    return x_out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input audio folder")
    parser.add_argument("--label", "-l", required=True, help="Label CSV file")
    parser.add_argument("--model", "-m", required=True, help="Path to .keras model file")
    parser.add_argument("--loss", "-lo", choices=["categorical", "cartesian"], default="categorical")
    parser.add_argument("--convert", "-c", dest="do_convert", action="store_true", help="Enable ACN to FUMA")
    parser.set_defaults(do_convert=False)
    args = parser.parse_args()

    # Register custom layers globally for Keras model deserialization
    get_custom_objects().update({
        "LSTM": LSTM,
        "Bidirectional": Bidirectional,
        "Dense": Dense,
        "Dropout": Dropout,
        "Conv2D": Conv2D,
        "BatchNormalization": BatchNormalization,
        "MaxPooling2D": MaxPooling2D,
        "Reshape": Reshape,
        "TimeDistributed": TimeDistributed,
        "Input": Input
    })

    # Load model safely
    print("Loading model architecture...")
    model = build_cartesian_model()
    print("Loading weights from:", args.model)
    saved_model = load_model(args.model, compile=False)
    model.set_weights(saved_model.get_weights())
    print("Model ready.")

    # Read test data
    with open(args.label, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        test_data = list(reader)

    feature_tensor, labels = [], []
    for line in test_data:
        wavpath = os.path.join(args.input, line[0])
        if not wavpath.endswith('.wav'):
            wavpath += '.wav'
        fs, x = read(wavpath)
        x = x.T
        if args.do_convert:
            print("converting ACN input to FUMA...")
            x = acn2fuma(x)

        nChannel, nSmp = x.shape
        lFrame = 1024
        hop = lFrame // 2
        lSentence = nSmp // hop
        x_f = np.empty((nChannel, lSentence, lFrame // 2 + 1), dtype=complex)
        for iChannel in range(nChannel):
            x_f[iChannel] = dsp.stft(x[iChannel], lWindow=lFrame)

        x_f = x_f[np.newaxis, :, :25, :]
        feature = getNormalizedIntensity(x_f)
        label = [float(x) for x in line[4:6]]
        labels.append(label)
        feature_tensor.append(feature)

    inputFeat_f = np.vstack(feature_tensor)
    print("feature shape:", inputFeat_f.shape)

    pred_frames = model.predict(inputFeat_f)
    pred_avg = np.mean(pred_frames, axis=1)

    el_grid, az_grid = grid.makeDoaGrid(10)
    tol = 20
    errors = []

    for i, prediction in enumerate(pred_avg):
        if args.loss == "cartesian":
            direction = prediction / np.linalg.norm(prediction)
            azi = math.atan2(-direction[1], -direction[0])
            ele = math.acos(direction[2])
            error = tensor_angle(np.array([azi, ele]), np.array(labels[i]))
        else:
            peaks, iPeaks = grid.peaksOnGrid(prediction, el_grid, az_grid, tol)
            idx = np.argmax(peaks)
            predIdx = iPeaks[idx]
            ele, azi = el_grid[predIdx], az_grid[predIdx]
            azi, ele = toLOCATA(azi, ele)
            error = tensor_angle(np.deg2rad([azi, ele]), np.array(labels[i]))

        errors.append(error)
        print('predicted = ({:.4}, {:.4}), true = ({:.4}, {:.4}), error = {:.4}'.format(
            np.rad2deg(azi), np.rad2deg(ele),
            np.rad2deg(labels[i][0]), np.rad2deg(labels[i][1]),
            np.rad2deg(error)))

    angle_observations = np.array([5, 10, 15])
    angle_cnts = np.array([(np.rad2deg(errors) <= deg).sum() for deg in angle_observations])
    angle_accuracy = angle_cnts / len(errors)
    for deg, acc in zip(angle_observations, angle_accuracy):
        print(f"accuracy/deg{deg}: {acc * 100:.4f}%")
    print(f"average error: {np.mean(np.rad2deg(errors)):.4f} degrees")
    print("{},{:.4f}%,{:.4f}%,{:.4f}%,{:.4f}".format(
        args.model,
        angle_accuracy[0] * 100,
        angle_accuracy[1] * 100,
        angle_accuracy[2] * 100,
        np.mean(np.rad2deg(errors))
    ))
