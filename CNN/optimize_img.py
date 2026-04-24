from catdog_cnn import path
from PIL import Image

import os

def resize_images(path):
    counter = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(root, file)
                with Image.open(img_path) as img:
                    img = img.convert('RGB').resize((128,128))
                    img.save(img_path)
                    counter += 1
    print(f"Converted files {counter}")

resize_images(path)