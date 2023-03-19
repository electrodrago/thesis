# cd BUILDCODE/thesis/model
import torch
from TheVSR import TheVSR
from SPyNet_arch import SpyNet
from prettytable import PrettyTable
from collections import OrderedDict
from torchvision.utils import flow_to_image
import cv2
import matplotlib.pyplot as plt
from utils.img_utils import img2tensor, tensor2img

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


model = SpyNet("spynet.pth")
img1 = cv2.imread("../sample/00000001.png")
img2 = cv2.imread("../sample/00000002.png")
img1 = img1 / 255.
img2 = img2 / 255.

img1 = img2tensor(img1)
img2 = img2tensor(img2)
img1.unsqueeze_(0)
img2.unsqueeze_(0)

flow = model(img1, img2)

flow_img = flow_to_image(flow)
flow_img.squeeze_(0)

plt.imshow(flow_img.permute(1, 2, 0))
plt.show()


# model = TheVSR(64, 9, 'spynet.pth')

# input = torch.randn((2, 14, 3, 64, 64))

# # hqs, lqs = model(input, input)

# # print(hqs.shape, lqs.shape)
# model(input, False)
# count_parameters(model)

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
