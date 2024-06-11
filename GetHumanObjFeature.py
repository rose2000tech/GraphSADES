#Extract image features
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np

class ObjFeatures():
    #object_folder Replace with the path to the folder containing the image of the clipped object
    def __init__(self,object_folder='HICO_train2015_00000004_objects'):
        # Load cropped object image
        self.object_folder=object_folder

    def GetFeatures(self):
        # Convert image to tensor and normalize
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to the input size of the model
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        object_features = []

        # Loading pre-trained ResNet models
        model = resnet50(pretrained=True)
        model = model.eval()

        # Iterate over object images and extract features
        for filename in os.listdir(self.object_folder):
            if filename.endswith('.jpg'):
                object_path = os.path.join(self.object_folder, filename)
                object_image = Image.open(object_path)
                object_image = transform(object_image)
                object_image = object_image.unsqueeze(0)  # Add Batch Dimension
                with torch.no_grad():
                    features = model(object_image)
                object_features.append(features.numpy())

        # Stacking features into a NumPy array
        object_features = np.vstack(object_features)
        return object_features




