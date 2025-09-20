import torch
import torchvision
def create_mobile_netv2_model(num_classes:int):
    weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
    model = torchvision.models.mobilenet_v2(weights = weights)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = torch.nn.Linear(in_features=1280, out_features=num_classes)
    return model