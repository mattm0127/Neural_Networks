import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
import kagglehub

path = kagglehub.dataset_download("tongpython/cat-and-dog")

if torch.cuda.is_available():
    device = 'cuda'
elif torch.xpu.is_available():
    device = 'xpu'
else:
    device = 'cpu'

class CatDogCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.gpu_train_transform = v2.Compose([
            # Spacial Variations
            v2.RandomResizedCrop(128, scale=(0.7, 1.0)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(25),
            #  appearance variations
            v2.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5, 0.5, 0.5),
                                (0.5, 0.5, 0.5))
        ])

        self.gpu_test_transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

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
            #Block 4 - 16x16 -> 8x8
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8*8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        if self.training:
            x = self.gpu_train_transform(x)
        else:
            x = self.gpu_test_transform(x)

        x = self.features(x)
        x = self.classifier(x)
        
        return x


def get_data(root_path):
    
    cpu_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize((128, 128))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.ImageFolder(
        root=f"{root_path}/training_set/training_set",
        transform=cpu_transform
    )                     

    test_dataset = datasets.ImageFolder(
        root=f"{root_path}/test_set/test_set",
        transform=cpu_transform
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=128, 
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False
    )

    print(f"Classes: {train_dataset.classes}")
    print(f"Train Samples: {len(train_dataset)}")
    print(f"Validation Samples: {len(test_dataset)}")

    return train_loader, test_loader

def train(model, train_loader, criterion, optimizer, epochs=1):
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1} | Loss: {loss.item()}")
            test(model, test_loader)

def test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(f'\nTest set: Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)\n')

model = CatDogCNN().to(device)
train_loader, test_loader = get_data(path)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

train(model, train_loader, criterion, optimizer, epochs=30)
test(model, test_loader)
torch.save(model.state_dict(), 'catdog.pth')


