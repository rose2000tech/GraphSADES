#Detecting objects and humans and clipping
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
        self.boxes=[] #Detects all objects in the box
        self.labels=[] #Labeling of all detected objects
        # Loading pre-trained Faster R-CNN models
        self.model = fasterrcnn_resnet50_fpn(pretrained=True, rpn_post_nms_top_n_test=200,
                                        box_batch_size_per_image=128,
                                        box_score_thresh=0.9, box_nms_thresh=0.3, weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)

    #image_path Load the image to be detected
    def GetDet(self,image_path= 'image/HICO_train2015_00000024.jpg'):
        self.model.eval()

        # Functions for image conversion
        transform = transforms.Compose([transforms.ToTensor()])

        # Get the name of the original image file (excluding the extension)
        image_filename = os.path.splitext(os.path.basename(image_path))[0]
        # Create a folder named after the original image file name
        output_folder = f'{image_filename}_objects'
        # Check and delete existing folders
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder, exist_ok=True)

        image = Image.open(image_path)

        # Converting Images to PyTorch Tensor
        input_image = transform(image)

        # Object detection using models
        with torch.no_grad():
            prediction = self.model([input_image])

        # Extraction of test results
        self.boxes = prediction[0]['boxes'].cpu().numpy().astype(int)
        scores = prediction[0]['scores'].cpu().numpy()
        self.labels = prediction[0]['labels'].cpu().numpy()
        class_labels = metadata.coco_classes

        # Set thresholds to filter out low confidence detections
        # threshold = 0.5
        selected_boxes = self.boxes  # boxes[scores >= threshold]
        selected_labels = self.labels

        # Crop and save detected objects
        for i, (box, label) in enumerate(zip(selected_boxes, selected_labels)):
            x1, y1, x2, y2 = box
            cropped_object = image.crop((x1, y1, x2, y2))
            # pdb.set_trace()
            object_name = class_labels[label]
            object_filename = f'{object_name}_{i}.jpg'
            object_path = os.path.join(output_folder, object_filename)
            cropped_object.save(object_path) #Crop out the image and save it to a folder

            # Labeling of object names on the original drawing
            # draw = ImageDraw.Draw(image)
            # draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            # draw.text((x1, y1), object_name, fill='red')
            # del draw

        # plt.imshow(np.array(image))
        # currentAxis = plt.gca()
        # for box in selected_boxes:
        #     x1, y1, x2, y2 = box
        #     currentAxis.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=None, color='red'))
        # plt.show()


    def GetAdjMatrix(self,):
        # Category labels for humans and objects
        person_label = 1  # Assuming that the category label of the person is 1

        # Initialize the adjacency matrix
        num_objects = len(self.boxes)
        adjacency_matrix = torch.zeros((num_objects, num_objects))

        # Calculating spatial distances between people and other objects and constructing the adjacency matrix
        for i in range(num_objects):
            for j in range(num_objects):
                if i != j:  # Does not count the distance of the object from itself
                    if self.labels[i] == person_label:  # If it's a person
                        center_i = (self.boxes[i][0] + self.boxes[i][2]) / 2, (self.boxes[i][1] + self.boxes[i][3]) / 2
                        center_j = (self.boxes[j][0] + self.boxes[j][2]) / 2, (self.boxes[j][1] + self.boxes[j][3]) / 2
                        distance = torch.norm(torch.tensor(center_i) - torch.tensor(center_j))
                        adjacency_matrix[i, j] = distance
                        # print(f"人-{metadata.coco_classes[self.labels[j]]} distance is {distance}，(i,j)=({i},{j})")

        return adjacency_matrix









