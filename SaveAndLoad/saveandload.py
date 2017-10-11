import torch 
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


torch.manual_seed(1)

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2)+0.2*torch.rand(x.size())

x = Variable(x,requires_grad = False)
y = Variable(y,requires_grad = False)


def save():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )

    optimizer = torch.optim.SGD(net1.parameters(),lr = 0.5)
    loss_func = torch.nn.MSELoss()

    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(net1,'net.pkl') # method 1 
    torch.save(net1.state_dict(),'net.params.pkl') # method 2


# method 1
def restore_net():
    net2 = torch.load('net.pkl')
    prediction = net2(x)


# method 2
def restore_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )
    net3.load_state_dict(torch.load('net.params.pkl'))
    prediction = net3(x)



save()

restore_net()

restore_params()



