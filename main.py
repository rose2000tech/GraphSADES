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
import numpy as np  # NumPy

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

#Detecting objects in an image
objdet=ObjDet()
#Detecting objects in an image and cropping the image, caching it into a folder
objdet.GetDet(image_path= 'image/HICO_train2015_00000024.jpg')
adjacency_matrix=objdet.GetAdjMatrix()  #Get the adjacency matrix of the distance
objfeature=ObjFeatures(object_folder='HICO_train2015_00000024_objects')
features=objfeature.GetFeatures()#Getting distance for detecting object features
#print(features,adjacency_matrix)
print(torch.nonzero(adjacency_matrix).transpose(0, 1))
#Building PyTorch Geometric Data Objects, One Data One Map
y=torch.tensor([[0,1],[0,1],[1,0],[1,0]])
data = Data(x=torch.tensor(features), edge_index=torch.nonzero(adjacency_matrix).transpose(0, 1), edge_attr=None, y=y)
#G=to_networkx(data,to_undirected=True)
#visualize_graph(G,color=data.y)





feature_dim=len(features)
print(feature_dim)
#input_dim dimension of the weight matrix in the graph convolution layer, the input feature dimension of the graph convolution layer should match the input feature dimension of the weight matrix
model=GNNModel(input_dim=1000,hidden_dim=4 ,num_classes=2)

print(model)
print(data)



model.train()
criterion = torch.nn.CrossEntropyLoss() # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(),lr=0.01) # Define optimizer



def train(data) :
    optimizer.zero_grad()
    out,h= model(data) #It's a two-dimensional vector, mainly for us to draw a graph.
    #pdb.set_trace()
    loss = criterion(out, data.y.float()) # semi-supervised
    # (tensor([[0.5812, -0.4453],
    #          [0.5812, -0.4453],
    #          [0.5873, -0.4344],
    #          [0.5887, -0.4314]], grad_fn= < AddmmBackward0 >), tensor([1, 1, 2, 1]))
    loss.backward()
    optimizer.step()
    return loss

losses = []  # List for storing loss values
for epoch in range(401):
    loss = train(data)
    losses.append(loss.item())  # Convert PyTorch tensor to NumPy float using .item()
    print(f'Epoch {epoch+1}, Loss: {loss}')

# Creating and displaying lossy value images
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

print(data)

# Predicting Connectivity Relationships
model.eval()
with torch.no_grad():
    predictions = model(data)
    print(predictions)
    #predicted_connections = torch.argmax(predictions[0],dim=1)

    predicted_connections = tuple(torch.argmax(tensor, dim=1) for tensor in predictions)

# Connection relations for output prediction
print(predicted_connections)
