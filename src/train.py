
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
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
class CliArgs(Tap):
    """Command line arguments for training a model.

    Args:
        Tap (Tap): a class for argument parsing.
    """
    
    model: str 
    batch_size: int = 32
    epochs: int
    learning_rate: float = 0.001
    random_seed: int = 42
    data_ratio: float = 0.1




def main(args: CliArgs):
    """The main function for training the model.

    Args:
        args (CliArgs): Command line arguments for training.
    """
    
    pass
    if args.model == "TinyVGG":
        transform = Compose([
            Resize((224, 224)),
            ToTensor(),
        ])
    elif args.model == "ResNet50":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif args.model == "EfficientNet":
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


    if args.model == "TinyVGG":
        model = TinyVGG(input_shape=(3,224,224), hidden_units = 10, output_shape=len(class_names))
    elif args.model == "ResNet50":
        model = create_resnet_50_model(num_classes = len(class_names))
    elif args.model == "EfficientNet":
        model = create_efficient_net_model(num_classes = len(class_names))
    else:
        model = create_mobile_netv2_model(num_classes = len(class_names))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    log_dir = f"runs/{args.model}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)


    # Train the model
    train(model, train_loader, test_loader, optimizer, loss_fn, device = device, epochs=args.epochs, writer = writer)
    model_name = (f"{args.model}_model.pth - {args.epochs} epochs, {args.learning_rate} lr, {args.batch_size} batch size, {args.data_ratio} data ratio")
    torch.save(model.state_dict(), f"{model_name}")
    print(f"Model saved to {model_name}")
    writer.close()
    


    

if __name__ == "__main__":
    args = CliArgs().parse_args()
    main(args)



