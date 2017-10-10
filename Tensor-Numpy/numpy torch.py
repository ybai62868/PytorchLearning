import torch 
import numpy as np

np_data = np.arange(6).reshape((2,3))

torch_data = torch.from_numpy(np_data)


tensor2array = torch_data.numpy()

print (
	'\nnumpy\n', np_data,'\n'
	'\ntorch\n', torch_data,'\n',
	'\ntensor2array\n',tensor2array,
	)
