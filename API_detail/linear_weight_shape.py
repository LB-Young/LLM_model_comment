import torch.nn as nn
l1 = nn.Linear(in_features=14, out_features=10)
print(l1.weight.shape)
# torch.Size([10, 14])