#检测物体和人，并剪裁
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn,FasterRCNN_ResNet50_FPN_Weights
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import metadata
import pdb

class ObjDet():
    def __init__(self):
        self.boxes=[] #检测出所有物体的方框
        self.labels=[] #所有被检测出物体标签
        # 加载预训练的 Faster R-CNN 模型
        self.model = fasterrcnn_resnet50_fpn(pretrained=True, rpn_post_nms_top_n_test=200,
                                        box_batch_size_per_image=128,
                                        box_score_thresh=0.9, box_nms_thresh=0.3, weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)

    #image_path 加载待检测的图像
    def GetDet(self,image_path= 'image/HICO_train2015_00000024.jpg'):
        self.model.eval()

        # 用于图像转换的函数
        transform = transforms.Compose([transforms.ToTensor()])

        # 获取原图文件名（不包括扩展名）
        image_filename = os.path.splitext(os.path.basename(image_path))[0]
        # 创建以原图文件名命名的文件夹
        output_folder = f'{image_filename}_objects'
        # 检查并删除已存在的文件夹
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder, exist_ok=True)

        image = Image.open(image_path)

        # 将图像转换为 PyTorch 张量
        input_image = transform(image)

        # 使用模型进行物体检测
        with torch.no_grad():
            prediction = self.model([input_image])

        # 提取检测结果
        self.boxes = prediction[0]['boxes'].cpu().numpy().astype(int)
        scores = prediction[0]['scores'].cpu().numpy()
        self.labels = prediction[0]['labels'].cpu().numpy()
        class_labels = metadata.coco_classes

        # 设置阈值，以过滤掉低置信度的检测结果
        # threshold = 0.5
        selected_boxes = self.boxes  # boxes[scores >= threshold]
        selected_labels = self.labels

        # 剪裁并保存检测到的物体
        for i, (box, label) in enumerate(zip(selected_boxes, selected_labels)):
            x1, y1, x2, y2 = box
            cropped_object = image.crop((x1, y1, x2, y2))
            # pdb.set_trace()
            object_name = class_labels[label]
            object_filename = f'{object_name}_{i}.jpg'
            object_path = os.path.join(output_folder, object_filename)
            cropped_object.save(object_path) #剪裁出的图片，保存到文件夹

            # 在原图上标注物体名称
            # draw = ImageDraw.Draw(image)
            # draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            # draw.text((x1, y1), object_name, fill='red')
            # del draw

        # 显示原图和检测结果
        # plt.imshow(np.array(image))
        # currentAxis = plt.gca()
        # for box in selected_boxes:
        #     x1, y1, x2, y2 = box
        #     currentAxis.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=None, color='red'))
        # plt.show()


    def GetAdjMatrix(self,):
        # 人类和物体的类别标签
        person_label = 1  # 假设人的类别标签为1

        # 初始化邻接矩阵
        num_objects = len(self.boxes)
        adjacency_matrix = torch.zeros((num_objects, num_objects))

        # 计算人与其他物体的空间距离并构建邻接矩阵
        for i in range(num_objects):
            for j in range(num_objects):
                if i != j:  # 不计算对象与自身的距离
                    if self.labels[i] == person_label:  # 如果是人
                        center_i = (self.boxes[i][0] + self.boxes[i][2]) / 2, (self.boxes[i][1] + self.boxes[i][3]) / 2
                        center_j = (self.boxes[j][0] + self.boxes[j][2]) / 2, (self.boxes[j][1] + self.boxes[j][3]) / 2
                        distance = torch.norm(torch.tensor(center_i) - torch.tensor(center_j))
                        adjacency_matrix[i, j] = distance
                        # print(f"人-{metadata.coco_classes[self.labels[j]]}的距离{distance}，(i,j)=({i},{j})")

        return adjacency_matrix









