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
        return f"🔥 Incendiu forestier detectat!\nConfidenta: {confidence}%"
    else:
        return f"✅ Nu s-a detectat niciun incendiu!\nConfidenta: {100 - confidence}%"


with gr.Blocks() as demo:
    gr.Markdown("### 🔥 Fire Detection Model 🔥")
    gr.Markdown("📸 Încărcați o imagine pentru a verifica dacă conține un incendiu forestier.\n")
    
    # Definim un input de imagine
    image_input = gr.Image(type="numpy", label="Încărcați imaginea")
    
    run_button = gr.Button("Run")
    
    # Definim output-ul (un textbox pentru rezultat)
    output_text = gr.Textbox(label="Rezultatul Detecției", interactive=False)
    
    # Adăugăm un buton de Ru
    
    # Funcția care va fi apelată la apăsarea butonului
    run_button.click(fn=predict, inputs=image_input, outputs=output_text)

demo.launch()

# # Rulează aplicația Gradio
# print(f"Using device: {torch.device('cuda:3')}")
# print(f"CUDA available: {torch.cuda.is_available()}")

# iface.launch()
