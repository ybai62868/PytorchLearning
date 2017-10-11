import torch
import torchvision # Not absolutely
from torch.autograd import Variable # type
import torch.nn.functional as F # Function
import matplotlib.pyplot as plt # plot

x = torch.unsqueeze (torch.linspace(-1,1,100),dim = 1 )

#print (x.size())

y = x.pow(2) + 0.2*torch.rand(x.size())

x = Variable(x)
y = Variable(y)

plt.scatter(x.data.numpy(),y.data.numpy())

# plt.show()


class Net( torch.nn.Module ):
    # initialize the structure of the neural network
    def __init__(self,n_feature, n_hidden, n_output):
        super(Net,self).__init__() # inherit
        self.hidden = torch.nn.Linear( n_feature, n_hidden )
        self.predict = torch.nn.Linear( n_hidden, n_output ) 


    # forward calculation
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x 


net = Net(1,10,1)

print (net) 

plt.ion()
plt.show()


# optimizer
optimizer = torch.optim.SGD(net.parameters(),lr = 0.2)
loss_func = torch.nn.MSELoss()

for t in range(1000):
    prediction = net(x)

    loss = loss_func(prediction,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t%5==0:
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw = 5 )
        plt.text(0.5,0,'Loss = %.4f' % loss.data[0],fontdict = {'size':20,'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()

