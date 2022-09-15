# @Time    : 2022.9.15
# @Author  : RangerOnMars
# @File    : dataset_sampler.py

from tqdm import tqdm
import os
from shutil import copyfile

src_dir = "Zooniverse_audio_1sec/audio_1sec"
dst_dir = "Zooniverse_audio_1sec_refined"
src_csv = "audio_1sec.csv"
dst_csv = "data.csv"

print("="*50)
print("Analyzing...")
with open(src_csv,"r+") as src:
    with open(dst_csv,'w') as dst:
        for line in tqdm(src.readlines()):
            #Analyze Start
            title, str_y, str_n, str_unknown,subtitle = line.split(",")
            
            y = int(str_y)
            n = int(str_n)
            unknown = int(str_unknown)
            
            #Make sure the data is available
            if ((y + n + unknown) < 5 or (max(1, min(y,n,unknown)) / max(y,n,unknown) > 0.3)):
                continue
            
            dst.write(line)
            copyfile(os.path.join(src_dir, title + ".wav"), os.path.join(dst_dir,  title + ".wav"))
print("Done")