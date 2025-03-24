import torch
import torchvision.transforms as transforms
from PIL import Image
import pytorch_lightning as pl
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import Accuracy
import gradio as gr

class FireClassifier(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 1)  # Binary classification
        self.criterion = nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy(task='binary')
        self.lr = lr
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

# Încarcă modelul din checkpoint
checkpoint_path = "checkpoints/resnet18_forest_dataset/fire-classifier-epoch=00-val_acc=1.00.ckpt"
model = FireClassifier.load_from_checkpoint(checkpoint_path)
model.eval()
model.to(torch.device("cuda:3"))

# Definirea transformărilor pentru imagine
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image):
    # Aplică transformările pe imagine
    image = transform(image)
    image = image.unsqueeze(0)  # Adaugă batch dimension
    image = image.to(torch.device("cuda:3"))
    
    # Face predicția
    with torch.no_grad():
        output = model(image)
        probability = torch.sigmoid(output).item()  # Probabilitate
    
    confidence = round(probability * 100, 2)  # Convertim la procentaj
    
    if probability > 0.5:
        return f"🔥 Fire detected!\nConfidence: {confidence}%"
    else:
        return f"✅ No fire detected\nConfidence: {100 - confidence}%"

# Creează interfața Gradio
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="🔥 Fire Detection Model 🔥",
    description="📸 Încărcați o imagine pentru a verifica dacă conține foc.\n"
)

# Rulează aplicația Gradio
print(f"Using device: {torch.device('cuda:3')}")
print(f"CUDA available: {torch.cuda.is_available()}")

iface.launch()
