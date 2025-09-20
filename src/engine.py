import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, device: torch.device) -> tuple[float, float]:
    """To perform a single train step in a training process

    Args:
        model (torch.nn.Module): model to train
        dataloader (torch.utils.data.DataLoader): dataloader for training data
        optimizer (torch.optim.Optimizer): optimizer for model parameters
        loss_fn (torch.nn.Module): loss function to optimize
        device (torch.device): device to perform training on (CPU or GPU)

    Returns:
        tuple[float, float]: average loss and accuracy for the training step
    """

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        probabilities = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(probabilities, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def test_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, device: torch.device):
    """Summary: To perform a single test step in a training process

    Args:
        model (torch.nn.Module): model to evaluate
        dataloader (torch.utils.data.DataLoader): dataloader for test data
        loss_fn (torch.nn.Module): loss function to optimize
        device (torch.device): which device to perform testing on (CPU or GPU)

    Returns:
        tuple[float, float]: average loss and accuracy for the test step
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * images.size(0)
            probabilities = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(probabilities, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def train(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          test_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          device: torch.device,
          epochs: int, writer: SummaryWriter) -> dict:
    """Train and evaluate a model for a number of epochs

    Args:
        model (torch.nn.Module): model to train
        train_loader (torch.utils.data.DataLoader): dataloader for training data
        test_loader (torch.utils.data.DataLoader): dataloader for test data
        optimizer (torch.optim.Optimizer): optimizer for model parameters
        loss_fn (torch.nn.Module): loss function to optimize
        device (torch.device): which device to perform training on
        epochs (int): number of epochs to train for
        writer (SummaryWriter): TensorBoard SummaryWriter for saving logs

    Returns:
        dict: a dictionary containing the training history
    """
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in range(epochs):
        train_loss, train_acc = train_step(model, train_loader, optimizer, loss_fn, device)
        test_loss, test_acc = test_step(model, test_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Test', test_acc, epoch)

    return history  