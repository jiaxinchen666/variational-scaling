import torch.nn as nn
from torch.autograd import Variable
import torch

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class Convnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

class generator(nn.Module):
    def __init__(self,input,output):
        super(generator, self).__init__()
        self.output=output
        self.fc1 = nn.Linear(input, 600)
        #self.fc2=nn.Linear(1000,1000)
        #self.fc3=nn.Linear(1000,1000)
        self.fc4 = nn.Linear(600, 2*self.output)

        self.relu=nn.ReLU()

    def reparameterize(self, mean, var):
        if self.training:
            eps = Variable(var.data.new(var.size()).normal_())
            return eps.mul(var).add(mean)
        else:
            return mean

    def encoder(self,x,output):
        #rep=self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(x))))))
        rep=self.relu(self.fc1(x))
        logmean,logvar=torch.split(self.fc4(rep),output,dim=-1)
        mean = logmean.exp()
        #var = logvar.exp()
        var=torch.ones(logvar.size()).cuda()*1.0
        return mean,var

    def forward(self,x):
        mean,var=self.encoder(x,self.output)
        return self.reparameterize(mean,var),mean,var