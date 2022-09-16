# @Time    : 2022.9.15
# @Author  : RangerOnMars
# @File    : dataset_sampler.py

from sympy import false
from tqdm import tqdm
import os
import csv
from shutil import copyfile

data_dir = "humbugdb_neurips_2021_1"
src_csv = "neurips_2021_zenodo_0_0_1.csv"
dst_csv = "data.csv"



print("="*50)
print("Walking through all .wav files...")
wav_idxs = []
for root, dirs, files in os.walk(data_dir, topdown=False):
    for file in tqdm(files):
        if file.endswith(".wav"):
            wav_idxs.append(file.replace(".wav", ""))
print("Done")
    
print("="*50)
print("Generating new .csv file")
is_header_skipped = False
with open(src_csv,"r+") as src:
    with open(dst_csv,'w') as dst:
        csvwriter = csv.writer(dst, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for data in tqdm(csv.reader(src, delimiter=",", quotechar='"')):
            # print(data)
            if (not is_header_skipped):
                is_header_skipped = True
                csvwriter.writerow(data)
            else:
                #Analyze Start
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
                
                if (wav_idxs.count(id) == 0):
                    continue
                
                csvwriter.writerow(data)
            
print("Done")