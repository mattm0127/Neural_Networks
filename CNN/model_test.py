from catdog_cnn import CatDogCNN
import torch
from torchvision.transforms import v2
from PIL import Image

device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')

inference_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.uint8, scale=True),
    v2.Resize((128, 128)),
])

model = CatDogCNN().to(device)

model.load_state_dict(torch.load('catdog.pth', weights_only=True))
model.eval()

def predict(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = inference_transform(img)
    img_tensor = img_tensor.unsqueeze(0).to(device)


    with torch.no_grad(): # Don't track gradients (saves memory)
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    classes = ['cat', 'dog']
    result = classes[predicted.item()]
    print(f"Prediction: {result} ({confidence.item()*100:.2f}%)")


predict(r"")