import torch
from torch import nn,device
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets
from torchvision.transforms import v2

#define neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(in_features=28*28,out_features=512)
        self.relu1=nn.ReLu()
        self.linear2=nn.Linear(in_features=512,out_features=512)
        self.relu2=nn.ReLU()
        self.linear3=nn.Linear(in_features=512,out_features=10)

  def forward(self, x):
        x=self.flatten(x)
        x=self.linear1(x)
        x=self.relu1(x)
        x=self.linear2(x)
        x=self.relu2(x)
        logits=self.linear3(x)
        out=logits
        return out

def load_data():
    #download train dataset
        training_data=datasets.FashionMNIST(root='./data',train=True,download=True,
transform=v2.Compose([v2.ToImage(),v2.ToDtype(torch.float32,scale=True)]))
    #download test data
         test_data=datasets.FashionMNIST(root='./data',train=False,download=True,
    transform=v2.Compose([v2.ToImage(),v2.ToDtype(torch.float32,scale=True)]),)
        return training_data,test_data
