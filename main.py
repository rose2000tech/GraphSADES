import torch
import torch_geometric
from torch_geometric.data import Data, DataLoader
from GetHumanObjFeature import ObjFeatures
from GNNModel import GNNModel
from imgdet import ObjDet
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
#from torchsummary import summary
from torchinfo import  summary
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import pdb
import numpy as np  # 导入 NumPy

def visualize_graph(G, color):
    plt.figure(figsize=(7,7))
    plt.xticks()
    plt.yticks([])
    nx.draw_networkx(G,pos=nx.spring_layout(G, seed=42),with_labels=False,node_color=color,cmap="Set2")
    plt.show()

def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:,0],h[:, 1],s=140,c=color,cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch:{epoch},Loss: {loss.item():.4f}', fontsize=16)
    plt.show()

#检测图像中的物体
objdet=ObjDet()
#检测图像中的物体，并把图像裁剪，缓存到文件夹中
objdet.GetDet(image_path= 'image/HICO_train2015_00000024.jpg')
adjacency_matrix=objdet.GetAdjMatrix()  #获取距离的邻接矩阵
objfeature=ObjFeatures(object_folder='HICO_train2015_00000024_objects')
features=objfeature.GetFeatures()#获取距离的检测物体特征
#print(features,adjacency_matrix)
print(torch.nonzero(adjacency_matrix).transpose(0, 1))
#构建 PyTorch Geometric 数据对象,一个Data一张图
y=torch.tensor([[0,1],[0,1],[1,0],[1,0]])
data = Data(x=torch.tensor(features), edge_index=torch.nonzero(adjacency_matrix).transpose(0, 1), edge_attr=None, y=y)
#G=to_networkx(data,to_undirected=True)
#visualize_graph(G,color=data.y)





feature_dim=len(features)
print(feature_dim)
#input_dim图卷积层中的权重矩阵的维度,图卷积层的输入特征维度应与权重矩阵的输入特征维度匹配
model=GNNModel(input_dim=1000,hidden_dim=4 ,num_classes=2)

print(model)
print(data)



model.train()
criterion = torch.nn.CrossEntropyLoss() # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(),lr=0.01) # Define optimizer



def train(data) :
    optimizer.zero_grad()
    out,h= model(data) #是两维向量，主要是为了咱们画个图
    #pdb.set_trace()
    loss = criterion(out, data.y.float()) # semi-supervised
    # (tensor([[0.5812, -0.4453],
    #          [0.5812, -0.4453],
    #          [0.5873, -0.4344],
    #          [0.5887, -0.4314]], grad_fn= < AddmmBackward0 >), tensor([1, 1, 2, 1]))
    loss.backward()
    optimizer.step()
    return loss

losses = []  # 用于存储损失值的列表
for epoch in range(401):
    loss = train(data)
    losses.append(loss.item())  # 使用.item()将PyTorch张量转换为NumPy float
    print(f'Epoch {epoch+1}, Loss: {loss}')

# 创建并显示损失值图像
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

print(data)

# 预测连接关系
model.eval()
with torch.no_grad():
    predictions = model(data)
    print(predictions)
    #predicted_connections = torch.argmax(predictions[0],dim=1)

    predicted_connections = tuple(torch.argmax(tensor, dim=1) for tensor in predictions)

# 输出预测的连接关系
print(predicted_connections)
