import gradio as gr
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from models.resnet_50 import create_resnet_50_model
from torchvision.models import EfficientNet_B2_Weights
from utils import get_cwd
import os


class_names = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']
class_names_2 = ['organic','recycleable']
model = create_resnet_50_model(num_classes=len(class_names))
model.load_state_dict(torch.load('c:/Users/admin/Desktop/classification pj/ResNet50_model.pth', weights_only=True))
model.eval()
model_2 = create_resnet_50_model(num_classes=len(class_names_2))
model_2.load_state_dict(torch.load('c:/Users/admin/Desktop/classification pj/ResNet50_model1.pth - 5 epochs, 0.001 lr, 32 batch size, 0.1 data ratio', weights_only=True))
model_2.eval()
weights = torchvision.models.ResNet50_Weights.DEFAULT
transform = weights.transforms()

def predict(image: Image.Image):
    width, height = image.size
    with torch.inference_mode():
        img = transform(image).unsqueeze(0)
        output = model(img)
        pred = torch.argmax(output, dim=1).item()
        pred_class = class_names[pred]
        output_2 = model_2(img)
        pred_2 = torch.argmax(output_2, dim=1).item()
        pred_class_2 = class_names_2[pred_2]
    return pred_class, pred_class_2

demo = gr.Interface(
    fn = predict,
    inputs = gr.Image(type="pil"),
    outputs = [gr.Label("text"),
               gr.Label("text"),
              ]
)


if __name__ == "__main__":
    demo.launch(share=True)