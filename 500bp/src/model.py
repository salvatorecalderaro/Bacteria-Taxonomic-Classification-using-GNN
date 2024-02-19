from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch
from time import perf_counter as pc
from datetime import timedelta
import torch.nn.functional as F
from tqdm import tqdm
from time import perf_counter as pc

class GCN(torch.nn.Module):
    def __init__(self,in_features,hidden_channels,num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels,num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.lin(x)

        return x
    

def train_net(device,net,trainloader,epochs,lr):
    optimizer = torch.optim.Adam(net.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    net=net.to(device)
    start=pc()
    print("Training....")
    for epoch in tqdm(range(epochs)):
        net.train()
        epoch_loss=0
        for i,data in enumerate(trainloader):
            # Iterate in batches over the training dataset.
            data=data.to(device)
            out = net(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
            epoch_loss += loss.detach().item()
        epoch_loss /= (i + 1)
        #print('Epoch {}, loss {:.4f}'.format(epoch+1, epoch_loss))
    end=pc()
    t=end-start
    total_time=timedelta(seconds=t)
    print("Training time:" ,str(total_time))
    return net,t


def predict(device,net,testloader):
    y_true = []
    y_pred = []
    net.eval()
    net.to(device)
    start=pc()
    with torch.no_grad():
        for data in testloader:
            data=data.to(device)
            labels=data.y.tolist()
            y_true+=labels
            out = net(data.x, data.edge_index, data.batch)
            proba=F.softmax(out,dim=1)
            pred = proba.argmax(dim=1, keepdim=True).tolist()
            y_pred+=pred
    end=pc()
    elapsed_time=end-start
    y_true=torch.tensor(y_true)
    y_pred=torch.tensor(y_pred).reshape(-1)
    return y_true,y_pred,elapsed_time