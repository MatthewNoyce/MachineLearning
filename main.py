import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import pandas as pd
import csv
import os 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    with open("labels_train.csv") as f:
        reader = csv.reader(f)
        data = list(reader)
        data = data[1:]


    class CarsDataset(Dataset):
        def __init__(self, data, transform=None):
            self.img_dir = "images/images/"
            self.labels = data
            self.transform = transform

        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            img_path = self.labels[idx][0]
            image = read_image(os.path.join(self.img_dir, img_path))
            label = self.labels[idx][5]
            bbox_xxyy = [self.labels[idx][1],self.labels[idx][2],self.labels[idx][3],self.labels[idx][4]]
            if self.transform:
                image = self.transform(image)
            return image, label, bbox_xxyy
        
    train_dataset = CarsDataset(data)
    train_datloader = DataLoader(train_dataset, batch_size=16, num_workers=4, shuffle=False)
    for i in enumerate(train_datloader):
        print(len(i))
        
