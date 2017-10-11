# coding=utf-8
import torch
import torch.utils.data as Data

BATCH_SIZE = 8

x = torch.linspace(1,10,10)
y = torch.linspace(10,1,10)

torch_dataset = Data.TensorDataset(data_tensor = x,target_tensor = y)

loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True, # 打乱顺序
    num_workers = 2
)

for epoch in range(4):
    for step, (batch_x,batch_y) in enumerate (loader ):
        # training...
           print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())
