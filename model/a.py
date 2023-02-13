# cd BUILDCODE/thesis/model
import torch
from TheVSR import TheVSR


model = TheVSR(64, 15, 'spynet.pth')

input = torch.randn((2, 15, 3, 64, 64))

hqs, lqs = model(input)

print(hqs.shape, lqs.shape)
