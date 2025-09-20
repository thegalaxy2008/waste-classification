import torch
import torchvision

def create_efficient_net_model(num_classes:int):
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = torch.nn.Linear(in_features=1408, out_features=num_classes)
    return model