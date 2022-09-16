# @Time    : 2022.9.15
# @Author  : RangerOnMars
# @File    : converter.py

import cv2
import scipy.io.wavfile as wav
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt
import numpy as np
import time

import librosa

start = 0
new_sample_rate = 16000

sr,y = wav.read("humbugdb_neurips_2021_1/4243.wav")
y = y*1.0
new_signal = librosa.resample(y, sr, new_sample_rate)

for i in range(100):
    data = np.reshape(new_signal[start: start + 128],(1,128))

    t1 = time.time()
    transformer = GramianAngularField()
    img = transformer.transform(data)
    print(img)
    img = np.array((img + 1) * 127,dtype=np.uint8)
    t2 = time.time()
    print(t2 - t1)
    cv2.imshow("dst",img.reshape(128,128))
    cv2.waitKey(500)
    start+=128