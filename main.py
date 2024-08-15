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
    train_datloader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    #Train_dataloader gives classid, bbox coords



#Neural network class for class_id
    class Cnn(nn.Module):
        def __init__(self):
            super(Cnn,self).__init__()
    
            self.layer1 = nn.Sequential(
                nn.Conv2d(3,16,kernel_size=3, padding=0,stride=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            
            self.layer2 = nn.Sequential(
                nn.Conv2d(16,32, kernel_size=3, padding=0, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2)
                )
            
            self.layer3 = nn.Sequential(
                nn.Conv2d(32,64, kernel_size=3, padding=0, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            
            
            self.fc1 = nn.Linear(2304,512)
            self.fc2 = nn.Linear(512,512)
            self.fc3 = nn.Linear(512,5)
            self.tanh = nn.Tanh()

        def forward(self,x):

            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = out.view(out.size(0),-1)
            out = self.tanh(self.fc1(out))
            out = self.tanh(self.fc2(out))
            out = self.fc3(out)

            return out

