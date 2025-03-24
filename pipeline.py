import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Accuracy

# Define FireDataset class
class FireDataset(Dataset):
    def __init__(self, csv_file, root_dir, type="train", transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.type = type
        self.annotations = self.annotations[self.annotations["step"] == self.type]
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        # print(img_path)
        # try:
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # except:
            # print(img_path)
        # print(image)
        label = torch.tensor(int(self.annotations.iloc[index, 1]), dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label

# Define FireDataModule
class FireDataModule(pl.LightningDataModule):
    def __init__(self, csv_file, root_dir, batch_size=32, num_workers=4):
        super().__init__()
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def setup(self, stage=None):
        dataset_train = FireDataset(self.csv_file, self.root_dir, type="train", transform=self.transform)
        dataset_val = FireDataset(self.csv_file, self.root_dir, type="val", transform=self.transform)
        self.train_dataset = dataset_train
        self.val_dataset = dataset_val

    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)

# Define FireClassifier Model
class FireClassifier(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        # self.model = models.resnet101(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 1)  # Binary classification
        self.criterion = nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy(task='binary')
        self.lr = lr
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels.unsqueeze(1).float())
        acc = self.accuracy(outputs.sigmoid(), labels.unsqueeze(1).float())
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels.unsqueeze(1).float())
        acc = self.accuracy(outputs.sigmoid(), labels.unsqueeze(1).float())
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

# Training Pipeline
def train_model(csv_file, root_dir, batch_size=32, max_epochs=10):
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/resnet18_wildfire_dataset",  # Directory to save checkpoints
        filename="fire-classifier-{epoch:02d}-{val_acc:.2f}",  # Checkpoint filename format
        save_top_k=1,  # Saves top 3 models based on validation loss
        monitor="val_acc",  # Monitor validation loss
        mode="max",  # Save models with the lowest validation loss
        save_weights_only=False  # Save full model (architecture + optimizer state)
    )
    data_module = FireDataModule(csv_file, root_dir, batch_size)
    model = FireClassifier()
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator='gpu' if torch.cuda.is_available() else 'cpu', callbacks=[checkpoint_callback] )
    trainer.fit(model, data_module)

# Example usage:
os.environ["CUDA_VISIBLE_DEVICES"] ="0"
# train_model('datasets/forest_dataset_2.csv', 'datasets/forest_dataset', batch_size=32, max_epochs=10)
train_model('datasets/wildfire_dataset.csv', '/raid/bigdata/userhome/ionut.serban/sharedData/controlnet_mirpr/AIFireFighters/datasets/wildfire_dataset_r', batch_size=32, max_epochs=10)
