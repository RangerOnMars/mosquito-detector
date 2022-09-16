# @Time    : 2022.9.15
# @Author  : RangerOnMars
# @File    : converter.py

import cv2
import os
from paddle import dtype
import scipy.io.wavfile as wav
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt
import numpy as np
import time

import librosa
import csv
import warnings
from sklearn import datasets
from tqdm import tqdm

def wav2gasfs(file, target_sample_rate=16000, sample_span=128,max_batch=10):
    
    sr,y = wav.read(file)
    y = y * 1.0
    start_idx = []
    new_signal = []
    warnings.filterwarnings("ignore")
    new_signal = librosa.resample(y, sr, target_sample_rate)
    batch = len(new_signal) / sample_span
    
    assert batch > 1
    if batch < max_batch:
        batch = int(batch)
    else:
        batch = max_batch
    start_idx = np.linspace(0, len(new_signal) - sample_span, batch, dtype=int)
        
    results = []

    for i in range(batch):
        data = np.reshape(new_signal[start_idx[i]:start_idx[i]+128],(1,sample_span))

        transformer = GramianAngularField()
        img = transformer.transform(data)
        results.append(img)
    
    return results

src_dir = "humbugdb_neurips_2021_1"
dst_dir = "gaf"
label_csv = "data.csv"
basic_batch = 3

if __name__ == "__main__":
    is_header_skipped = False
    with open(label_csv,"r+") as csvfile:
        for data in tqdm(csv.reader(csvfile, delimiter=",", quotechar='"')):
            # print(data)
            if (not is_header_skipped):
                is_header_skipped = True
                continue
            else:
                [id,
                length,
                name,
                sample_rate,
                record_datetime,
                sound_type,
                species,
                gender,
                fed,
                plurality,
                age,
                method,
                mic_type,
                device_type,
                country,
                district,
                province,
                place,
                location_type] = data
                
                imgs = []
                if sound_type == "mosquito":
                    imgs = wav2gasfs(os.path.join(src_dir, id + ".wav"),max_batch=basic_batch)
                elif sound_type == "audio":
                    imgs = wav2gasfs(os.path.join(src_dir, id + ".wav"),max_batch=round(basic_batch * 2.5))
                else:
                    imgs = wav2gasfs(os.path.join(src_dir, id + ".wav"),max_batch=round(basic_batch * 20))
                for i in range(len(imgs)):
                    #Rescale from[-1,1] to [0,255]
                    img = np.array((imgs[i] + 1) * 127,dtype=np.uint8)
                    img = np.reshape(img, (128,128))
                    cv2.imshow("1",img)
                    cv2.waitKey(10)
                    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
                    cv2.imwrite(os.path.join(dst_dir,sound_type,id + "_"+ str(i) + ".png"),img)
                
                