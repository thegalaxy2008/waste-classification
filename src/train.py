
# changing hyperparamters using tap and -help
from data_setup import create_data_loaders
from tap import Tap
from torchvision.transforms import Compose, ToTensor
from model_builder import TinyVGG
from engine import train
import torch
import torch.nn as nn
from torchvision.transforms import Resize
class CliArgs(Tap):
    model_name: str = "resnet18"
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 0.001
    random_seed: int = 42
def main(args: CliArgs):
    pass
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
    ])
    train_loader, test_loader, class_names = create_data_loaders(train_dir = "dataset/garbage-dataset/train",
                        test_dir = "dataset/garbage-dataset/test", batch_size=args.batch_size, transforms=transform)
    model = TinyVGG(input_shape=(3, 224, 224), hidden_units=10, output_shape=len(class_names))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the model
    train(model, train_loader, test_loader, optimizer, loss_fn, device = device, epochs=args.epochs)
    torch.save(model.state_dict(), f"{args.model_name}_model.pth")


if __name__ == "__main__":
    args = CliArgs().parse_args()
    main(args)



