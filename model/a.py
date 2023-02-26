# cd BUILDCODE/thesis/model
import torch
from TheVSR import TheVSR
from SPyNet_arch import SpyNet
from prettytable import PrettyTable
from collections import OrderedDict



def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


model = TheVSR(64, 10, 'spynet.pth')

input = torch.randn((2, 14, 3, 64, 64))

# hqs, lqs = model(input, input)

# print(hqs.shape, lqs.shape)
model(input, False)
count_parameters(model)

# Load model for checking
# ckpt = torch.load("iter_7000.pth", map_location=lambda storage, loc: storage)

# ckpt_dict = OrderedDict()

# for i in ckpt['state_dict'].keys():
#     if 'generator_' not in i:
#         if 'step' in i:
#             continue
#         k_str = i.replace('generator.', '')
#         ckpt_dict[k_str] = ckpt['state_dict'][i]

# model.load_state_dict(ckpt_dict)
# print('Done')
