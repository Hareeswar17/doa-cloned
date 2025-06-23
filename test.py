# -*- coding: utf-8 -*-
"""
Modified test script to support loading .keras models using custom_objects.
"""

import csv
import os
import argparse
import math
import grid
import dsp
import numpy as np
from scipy.io.wavfile import read
from feature_extractor import getNormalizedIntensity

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import (LSTM, Bidirectional, Dense, Dropout,
                                     Conv2D, BatchNormalization, MaxPooling2D,
                                     Reshape, TimeDistributed, InputLayer)

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
    parser = argparse.ArgumentParser(prog='test',
                                     description="Script to test the DOA estimation system")
    parser.add_argument("--input", "-i", required=True, help="Directory of input audio files", type=str)
    parser.add_argument("--label", "-l", required=True, help="Path to label", type=str)
    parser.add_argument("--model", "-m", type=str, required=True, help="Model path")
    parser.add_argument("--loss", "-lo", type=str, choices=["categorical", "cartesian"],
                        default="categorical", help="Choose loss representation")
    parser.add_argument('--convert', "-c", dest='do_convert', action='store_true',
                        help='flag to enable ACN to FUMA conversion for FOA channel ordering')
    parser.set_defaults(do_convert=False)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print("input path {} non-exist, abort!".format(args.input))
        exit(1)

    labelpath = os.path.join(args.label)
    with open(labelpath, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)
        test_data = list(csvreader)

    _custom_objects = {
        "LSTM": LSTM,
        "Bidirectional": Bidirectional,
        "Dense": Dense,
        "Dropout": Dropout,
        "Conv2D": Conv2D,
        "BatchNormalization": BatchNormalization,
        "MaxPooling2D": MaxPooling2D,
        "Reshape": Reshape,
        "TimeDistributed": TimeDistributed,
        "InputLayer": InputLayer
    }

    model = load_model(args.model, compile=False, custom_objects=_custom_objects)

    feature_tensor = []
    labels = []
    for line in test_data:
        wavpath = os.path.join(args.input, line[0])
        if not wavpath.endswith('wav'):
            wavpath += '.wav'
        fs, x = read(wavpath)
        x = x.T
        if args.do_convert:
            print("converting ACN input to FUMA...")
            x = acn2fuma(x)
        nChannel, nSmp = x.shape
        lFrame = 1024
        nBand = lFrame // 2 + 1
        hop = lFrame // 2
        lSentence = nSmp // hop
        x_f = np.empty((nChannel, lSentence, nBand), dtype=complex)
        for iChannel in range(nChannel):
            x_f[iChannel] = dsp.stft(x[iChannel], lWindow=lFrame)
        x_f = x_f[np.newaxis, :, :25, :]
        feature = getNormalizedIntensity(x_f)
        label = [float(x) for x in line[4:6]]
        labels.append(label)
        feature_tensor.append(feature)

    inputFeat_f = np.vstack(feature_tensor)
    print("feature:{}".format(inputFeat_f.shape))

    predictProbaByFrame = model.predict(inputFeat_f)
    predictProbaByBatch = np.mean(predictProbaByFrame, axis=1)

    el_grid, az_grid = grid.makeDoaGrid(10)
    neighbour_tol = 20
    errors = []

    for i, prediction in enumerate(predictProbaByBatch):
        if args.loss == "cartesian":
            dir = prediction / np.linalg.norm(prediction)
            new_azi = math.atan2(-dir[1], -dir[0])
            new_ele = math.acos(dir[2])
            error = tensor_angle(np.array([new_azi, new_ele]), np.array(labels[i]))
        else:
            peaks, iPeaks = grid.peaksOnGrid(prediction, el_grid, az_grid, neighbour_tol)
            iMax = np.argmax(peaks)
            predIdx = iPeaks[iMax]
            el_pred = el_grid[predIdx]
            az_pred = az_grid[predIdx]
            new_azi, new_ele = toLOCATA(az_pred, el_pred)
            error = tensor_angle(np.deg2rad([new_azi, new_ele]), np.array(labels[i]))

        errors.append(error)
        print('predicted = ({:.4}, {:.4}), true = ({:.4}, {:.4}), error = {:.4}'.format(
            np.rad2deg(new_azi), np.rad2deg(new_ele),
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
