from pytorch_lightning import LightningDataModule
import pandas as pd
from torchvision import transforms
import cv2
import os
import torch

class FireDataset(LightningDataModule):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        if transform is None:
            transform = transforms.Compose([
                transforms.ToImage(),
                transforms.ToTensor(),  
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
            ])
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations["image"][index])
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        label = torch.tensor(int(self.annotations["label"][index]))

        if self.transform:
            image = self.transform(image)

        return image, label