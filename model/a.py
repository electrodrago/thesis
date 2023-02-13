# cd BUILDCODE/thesis/model
import torch
from TheVSR import TheVSR
from prettytable import PrettyTable


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


model = TheVSR(64, 7, 'spynet.pth')

input = torch.randn((2, 15, 3, 64, 64))

hqs, lqs = model(input)

print(hqs.shape, lqs.shape)

count_parameters(model)
