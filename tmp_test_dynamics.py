import numpy as np
from util.dynamics import masks_to_flows_gpu
masks = np.zeros((10,10),dtype=int)
masks[2:5,2:5]=1
masks[6:9,6:9]=2
flows = masks_to_flows_gpu(masks, device=None)
print('flows', flows.shape, flows.dtype, flows.min(), flows.max())
print(flows[:,2:5,2:5])
