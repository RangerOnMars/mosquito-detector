from random import sample
from torch.utils.data import Dataset
import cv2

class ImageDataset(Dataset):
    def __init__(self, label, transform=None, update_dataset=False):
        self.sample_list = []
        with open(label,"r") as file:
            for line in file.readlines():
                path, cls = line.split(" ")
                img = cv2.imread(path)
                grayscale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                self.sample_list.append([int(cls), grayscale])
    

    def __getitem__(self, index):
        img, label = self.sample_list[index]
        return img, label