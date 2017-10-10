import torch
import numpy as np

data = [[1,2],[3,4]]
#data = np.array(data)
tensor = torch.FloatTensor(data) # 32bit floating point

print(
	'\nnumpy:',np.matmul(data,data),
#	'\nnumpy:',data.dot(data),
	'\ntorch:',torch.mm(tensor,tensor),
	)
