#提取图像特征
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np

class ObjFeatures():
    #object_folder 替换为包含剪裁物体图像的文件夹路径
    def __init__(self,object_folder='HICO_train2015_00000004_objects'):
        # 加载剪裁的物体图像
        self.object_folder=object_folder

    def GetFeatures(self):
        # 转换图像为张量并进行标准化
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 将图像大小调整为模型的输入尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        object_features = []

        # 加载预训练的ResNet模型
        model = resnet50(pretrained=True)
        model = model.eval()

        # 遍历物体图像并提取特征
        for filename in os.listdir(self.object_folder):
            if filename.endswith('.jpg'):
                object_path = os.path.join(self.object_folder, filename)
                object_image = Image.open(object_path)
                object_image = transform(object_image)
                object_image = object_image.unsqueeze(0)  # 添加批次维度
                with torch.no_grad():
                    features = model(object_image)
                object_features.append(features.numpy())

        # 将特征堆叠成一个NumPy数组
        object_features = np.vstack(object_features)
        return object_features




