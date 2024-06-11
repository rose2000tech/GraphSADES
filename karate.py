import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import KarateClub
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import time
import pdb

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

dataset=KarateClub()
data=dataset[0]
pdb.set_trace()


class GCN(torch.nn.Module) :
    def __init__(self):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(dataset.num_features,4) # Simply define the input features and output features
        self.conv2 = GCNConv(4,4)
        self.conv3 = GCNConv(4,2)
        self.classifier = Linear(2,dataset.num_classes)

    def forward(self, x, edge_index):
        h=self.conv1(x,edge_index) # Input feature and adjacency matrix (note the format, the kind above)

        h = h. tanh()
        h = self.conv2(h,edge_index)
        h = h. tanh()
        h = self.conv3(h,edge_index)
        h = h. tanh()
        #分类层
        out = self.classifier(h)
        return out,h


model=GCN()
print(model)
# Calculate the number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params}")

criterion = torch.nn.CrossEntropyLoss() # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(),lr=0.01) # Define optimizer

def train(data) :
    optimizer.zero_grad()
    out,h = model(data.x,data.edge_index) #It's a two-dimensional vector, mainly for us to draw a graph.
    pdb.set_trace()
    loss = criterion(out[data.train_mask], data.y[data.train_mask]) # semi-supervised
    # (tensor([[-0.1800, 0.6862, 0.1598, 0.1413],
    #          [-0.1999, 0.6750, 0.1442, 0.1249],
    #          [-0.1940, 0.6523, 0.1278, 0.0997],
    #          [-0.1739, 0.7036, 0.1758, 0.1624]], grad_fn= < IndexBackward0 >), tensor([1, 3, 0, 2]))
    loss.backward()
    optimizer.step()
    return loss,h

for epoch in range(401):
    loss, h = train(data)
    if epoch %10 ==0:
        visualize_embedding(h, color=data.y,epoch=epoch, loss=loss)
        time.sleep(0.3)
