from random import sample
from torch.utils.data import Dataset
import cv2
from tqdm import tqdm
class ImageDataset(Dataset):
    def __init__(self, label, transform=None, update_dataset=False):
        self.sample_list = []
        print("Loading dataset from {}...".format(label))
        with open(label,"r") as file:
            for line in tqdm(file.readlines()):
                path, cls = line.split(" ")
                img = cv2.imread(path)
                grayscale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                gaf = (grayscale / 127.5) - 1
                # print(gaf)
                self.sample_list.append([gaf,int(cls)])
    

    def __getitem__(self, index):
        img, label = self.sample_list[index]
        return img, label
    
    def __len__(self):
        return len(self.sample_list)