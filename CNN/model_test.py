from catdog_cnn import CatDogCNN
import torch
from torchvision.transforms import v2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import math

if torch.cuda.is_available():
    device = "cuda"
elif torch.xpu.is_available():
    device = "xpu"
else:
    device = "cpu"

inference_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize((128, 128)),
    ]
)

model = CatDogCNN().to(device)

model.load_state_dict(
    torch.load(
        r"C:\Users\mattm\OneDrive\Desktop\PyTorch Models\CNN\catdog_93.pth",
        map_location=device,
        weights_only=True,
    )
)
model.eval()


def predict(model, img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = inference_transform(img)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():  # Don't track gradients (saves memory)
        output: torch.Tensor = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    if confidence < 0.85:
        result = "No Animal Detected"
    else:
        classes = ["cat", "dog"]
        result = classes[predicted.item()]
    print(f"Prediction: {result} | Confidence: {confidence.item() * 100:.2f}")


def display_layer(model, img_path, layer=0):

    activations = {}

    def hook_fn(model, input, output):
        # This grabs the output of the layer as it flies by
        activations["features"] = output.detach()

    target_layer = model.features[layer]
    target_layer.register_forward_hook(hook_fn)

    img = Image.open(img_path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    img_tensor = inference_transform(img)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        # Pass in a single image from your test set
        _ = model(img_tensor)

    act = activations["features"].squeeze()
    grid = int(math.sqrt(act.shape[0] / 2))
    fig, axes = plt.subplots(grid, grid, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < act.shape[0]:
            ax.imshow(act[i].cpu().numpy(), cmap="gist_gray")
            ax.axis("off")
    plt.suptitle(f"What the Model Sees At Layer {layer}")
    plt.show()


display_layer(
    model, r"C:\Users\mattm\Downloads\PXL_20250714_113507485.jpg", 0
)
