
# changing hyperparamters using tap and -help
from data_setup import create_data_loaders
from tap import Tap
from torchvision.transforms import Compose, ToTensor
from models import TinyVGG
from engine import train
import torch
import torch.nn as nn
from torchvision.transforms import Resize
from data_setup import create_partial_data_loaders
from torch.utils.data import DataLoader
from engine import train_step, test_step
from models.resnet_50 import create_resnet_50_model
import torchvision
from models.efficient_net import create_efficient_net_model
from models.mobile_netv2 import create_mobile_netv2_model
class CliArgs(Tap):
    """Command line arguments for training a model.

    Args:
        Tap (Tap): a class for argument parsing.
    """
    
    model_name: str 
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 0.001
    random_seed: int = 42
    data_ratio: float = 0.1




def main(args: CliArgs):
    """The main function for training the model.

    Args:
        args (CliArgs): Command line arguments for training.
    """
    
    pass
    if args.model_name == "TinyVGG":
        transform = Compose([
            Resize((224, 224)),
            ToTensor(),
        ])
    elif args.model_name == "ResNet50":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif args.model_name == "EfficientNet":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((288, 288)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    if args.data_ratio != 1:
        train_loader, test_loader, class_names = create_partial_data_loaders(train_dir = "dataset/garbage-dataset/train",
                        test_dir = "dataset/garbage-dataset/test", batch_size=args.batch_size, transforms=transform, ratio = args.data_ratio)
    else:
        train_loader, test_loader, class_names = create_data_loaders(train_dir = "dataset/garbage-dataset/train",
                        test_dir = "dataset/garbage-dataset/test", batch_size=args.batch_size, transforms=transform)

    
    if args.model_name == "TinyVGG":
        model = TinyVGG(input_shape=(3,224,224), hidden_units = 10, output_shape=len(class_names))
    elif args.model_name == "ResNet50":
        model = create_resnet_50_model(num_classes = len(class_names))
    elif args.model_name == "EfficientNet":
        model = create_efficient_net_model(num_classes = len(class_names))
    else:
        model = create_mobile_netv2_model(num_classes = len(class_names))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the model
    train(model, train_loader, test_loader, optimizer, loss_fn, device = device, epochs=args.epochs)
    torch.save(model.state_dict(), f"{args.model_name}_model.pth")
    print(f"Model saved to {args.model_name}_model.pth")
    


    

if __name__ == "__main__":
    args = CliArgs().parse_args()
    main(args)



