import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2

# Path to dataset for training
path = kagglehub.dataset_download("tongpython/cat-and-dog")

# Detect which device to perform training
if torch.cuda.is_available():
    device = "cuda"
elif torch.xpu.is_available():
    device = "xpu"
else:
    device = "cpu"


class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Image training transformations
        self.gpu_train_transform = v2.Compose(
            [
                # Spacial Variations
                v2.RandomResizedCrop(128, scale=(0.5, 1.0)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(25),
                #  appearance variations
                v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomErasing(p=0.2),
                v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Image testing transformations
        self.gpu_test_transform = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Feature Extraction
        self.features = nn.Sequential(
            # Block 1 - 128x128 -> 64x64
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 2 - 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 3 - 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 4 - 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 5 - 8x8 -> 4x4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor):
        if self.training:
            x = self.gpu_train_transform(x)
        else:
            x = self.gpu_test_transform(x)

        x = self.features(x)
        x = self.classifier(x)

        return x


def get_data(root_path: str):
    """Prepare data for training and testing

    Args:
        root_path (str): Path to the training/testing data

    Returns:
        (DataLoader, DataLoader): Training and Testing DataLoaders
    """

    # Transformations prior to GPU transformations
    cpu_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
        ]
    )

    train_dataset = datasets.ImageFolder(
        root=f"{root_path}/training_set/training_set", transform=cpu_transform
    )

    test_dataset = datasets.ImageFolder(
        root=f"{root_path}/test_set/test_set", transform=cpu_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )

    print(f"Classes: {train_dataset.classes}")
    print(f"Train Samples: {len(train_dataset)}")
    print(f"Validation Samples: {len(test_dataset)}")

    return train_loader, test_loader


def train(
    model: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epochs=1,
):
    top_accuracy = 0
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = (
                data.to(device, non_blocking=True),
                target.to(device, non_blocking=True),
            )
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        accuracy = test(model, test_loader)
        scheduler.step(accuracy)

        if accuracy > top_accuracy:
            top_accuracy = accuracy
            torch.save(model.state_dict(), "catdog.pth")

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1} | Loss: {loss.item()} | Accuracy: {accuracy:2f} | Learning Rate: {optimizer.param_groups[0]['lr']}"
            )


def test(model, test_loader):
    model.eval()
    total = len(test_loader.dataset)
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output: torch.Tensor = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    return correct / total


if __name__ == "__main__":
    model = CatDogCNN().to(device)
    train_loader, test_loader = get_data(path)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )
    train(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=50)
