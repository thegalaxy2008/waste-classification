import torch
import torchvision

def create_resnet_50_model(num_classes:int):
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights = weights)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes)
    return model

