from catdog_cnn import CatDogCNN
import torch
from torchvision.transforms import v2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')

inference_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.uint8, scale=True),
    v2.Resize((128, 128)),
])

model = CatDogCNN().to(device)

model.load_state_dict(torch.load('catdog.pth', weights_only=True))
model.eval()

def predict(img_path, model):
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


def display_layer(model, img_path, layer_num=0):


    activations = {}
    def hook_fn(model, input, output):
        # This grabs the output of the layer as it flies by
        activations['features'] = output.detach()

    target_layer = model.features[layer_num] 
    target_layer.register_forward_hook(hook_fn)

    img = Image.open(img_path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    img_tensor = inference_transform(img)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        # Pass in a single image from your test set
        _ = model(img_tensor)

    act = activations['features'].squeeze()
    fig, axes = plt.subplots(4, 8, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < act.shape[0]:
            ax.imshow(act[i].cpu().numpy(), cmap='magma')
            ax.axis('off')
    plt.suptitle("What the Model Sees (Deep Features)")
    plt.show()

display_layer(model, r"C:\Users\mattm\Downloads\PXL_20231111_185738680.jpg", )